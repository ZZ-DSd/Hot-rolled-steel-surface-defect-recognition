import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# =========================
# 基本配置
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.pth")

BASE_DATASET_DIR = os.path.join(BASE_DIR, "NEU", "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_annotations")

CLASSES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CNN（⚠️ 必须与训练时完全一致）
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 25 * 25, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.fmap = None
        self.grad = None

    def save_grad(self, grad):
        self.grad = grad

    def forward(self, x):
        x = self.features(x)
        self.fmap = x
        x.register_hook(self.save_grad)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# =========================
# 加载模型
# =========================
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =========================
# 图像预处理
# =========================
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# =========================
# 处理单张图像
# =========================
def process_image(img_path, save_path, threshold=0.5):
    image = Image.open(img_path).convert("L")
    x = transform(image).unsqueeze(0).to(device)

    # forward
    output = model(x)
    pred = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred].backward()

    # Grad-CAM
    weights = model.grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * model.fmap).sum(dim=1)
    cam = F.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (200, 200))

    # 二值化
    binary = (cam > threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 原图
    img_np = np.array(image.resize((200, 200)))
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    # 画缺陷框
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 200:
            cv2.rectangle(
                img_rgb,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

    # ✅ 用 PIL 保存（支持中文路径）
    Image.fromarray(img_rgb).save(save_path)



# =========================
# 批量生成
# =========================
print("🚀 开始批量生成缺陷标注图...")

for cls in CLASSES:
    input_dir = os.path.join(BASE_DATASET_DIR, cls)
    output_dir = os.path.join(OUTPUT_DIR, cls)

    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(input_dir)
              if f.lower().endswith((".bmp", ".jpg", ".png"))]

    print(f"{cls} 类别下读取到 {len(images)} 张图片")

    for img_name in tqdm(images, desc=f"Processing {cls}"):
        img_path = os.path.join(input_dir, img_name)
        save_path = os.path.join(output_dir, img_name.replace(".bmp", ".png"))

        process_image(img_path, save_path)

print("✅ 所有缺陷标注图已生成完成！")
