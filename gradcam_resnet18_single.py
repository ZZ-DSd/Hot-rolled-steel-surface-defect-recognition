import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# =========================
# 0. 基础配置
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
SAMPLES_PER_CLASS = 1      # 每类生成几张
SPLIT = "test"             # train / test

# =========================
# 1. 中文路径安全写图函数（关键）
# =========================
def cv2_imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    success, encoded_img = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError("cv2.imencode failed")
    encoded_img.tofile(path)

# =========================
# 2. 加载 ResNet18
# =========================
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 6)

model_path = os.path.join(BASE_DIR, "resnet18.pth")
assert os.path.exists(model_path), "❌ resnet18.pth 不存在"

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print("✅ ResNet18 模型加载成功")

# =========================
# 3. Grad-CAM Hook
# =========================
feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# ResNet18 最后一个卷积层
target_layer = model.layer4
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# =========================
# 4. 图像预处理
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# =========================
# 5. Grad-CAM 生成函数
# =========================
def generate_gradcam(img_path, save_path):
    feature_maps.clear()
    gradients.clear()

    # 中文路径安全读取
    with open(img_path, "rb") as f:
        image = Image.open(f).convert("L")

    input_tensor = transform(image).unsqueeze(0).to(device)

    output = model(input_tensor)
    class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    fmap = feature_maps[0]      # [1, C, H, W]
    grad = gradients[0]         # [1, C, H, W]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * fmap).sum(dim=1)
    cam = F.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (224, 224))

    img_np = np.array(image.resize((224, 224)))
    img_rgb = np.stack([img_np]*3, axis=-1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = 0.5 * heatmap + 0.5 * img_rgb
    overlay = overlay.astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ✅ 中文路径稳定保存
    cv2_imwrite_unicode(
        save_path,
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    print("✅ 保存完成:", save_path, "存在:", os.path.exists(save_path))

# =========================
# 6. 批量生成每类缺陷热力图
# =========================
out_root = os.path.join(BASE_DIR, "gradcam_results", "resnet18")
os.makedirs(out_root, exist_ok=True)

for cls in CLASSES:
    src_dir = os.path.join(BASE_DIR, "NEU", SPLIT, cls)
    save_dir = os.path.join(out_root, cls)

    print(f"\n📂 处理类别: {cls}")

    if not os.path.exists(src_dir):
        print("❌ 路径不存在，跳过")
        continue

    imgs = sorted(os.listdir(src_dir))[:SAMPLES_PER_CLASS]

    for img in imgs:
        img_path = os.path.join(src_dir, img)
        save_path = os.path.join(save_dir, img.replace(".bmp", "_cam.png"))
        generate_gradcam(img_path, save_path)

print("\n🎉 ResNet18 Grad-CAM 全部生成完成")
