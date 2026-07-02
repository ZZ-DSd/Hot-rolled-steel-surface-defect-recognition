import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# =========================
# 中文路径安全读写
# =========================
def pil_open_unicode(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.copy()

def cv2_imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    success, encoded_img = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError("cv2.imencode 失败")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    encoded_img.tofile(path)

# =========================
# 0. 基础配置
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
SAMPLES_PER_CLASS = 3
SPLIT = "train"

# =========================
# 1. CNN 模型
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
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 25 * 25, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# =========================
# 2. 加载模型
# =========================
model = SimpleCNN().to(device)
state_dict = torch.load(os.path.join(BASE_DIR, "cnn_model.pth"), map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()
print("✅ 模型加载成功")

# =========================
# 3. Hook
# =========================
feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

target_layer = model.features[6]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# =========================
# 4. 预处理
# =========================
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# =========================
# 5. Grad-CAM
# =========================
def generate_gradcam(img_path, save_path):
    feature_maps.clear()
    gradients.clear()

    image = pil_open_unicode(img_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)

    output = model(input_tensor)
    class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    fmap = feature_maps[0]
    grad = gradients[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * fmap).sum(dim=1)
    cam = F.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (200, 200))

    img_np = np.array(image.resize((200, 200)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.5 * heatmap + 0.5 * np.stack([img_np]*3, axis=-1)).astype(np.uint8)

    cv2_imwrite_unicode(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# =========================
# 6. 批量生成
# =========================
out_root = os.path.join(BASE_DIR, "gradcam_results", "cnn")

for cls in CLASSES:
    src_dir = os.path.join(BASE_DIR, "NEU", SPLIT, cls)
    save_dir = os.path.join(out_root, cls)

    print(f"\n📂 处理类别: {cls}")

    if not os.path.exists(src_dir):
        print("❌ 路径不存在，跳过")
        continue

    imgs = sorted(os.listdir(src_dir))[:SAMPLES_PER_CLASS]

    for img in imgs:
        generate_gradcam(
            os.path.join(src_dir, img),
            os.path.join(save_dir, img.replace(".bmp", "_cam.png"))
        )
        print("✅ 已生成:", img)

print("\n🎉 CNN Grad-CAM 全部生成完成")
