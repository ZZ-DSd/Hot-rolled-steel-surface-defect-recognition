import os
import time
import cv2
import torch
import numpy as np
import streamlit as st
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ===============================
# 0. 基础配置（解决中文路径）
# ===============================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]

st.set_page_config(
    page_title="热轧钢表面缺陷识别与检测系统",
    layout="wide"
)

st.title("🔍 热轧钢表面缺陷识别与检测系统")

# ===============================
# 1. 加载 ResNet18
# ===============================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load("resnet18.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ===============================
# 2. Grad-CAM Hook
# ===============================
features = []
grads = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    grads.append(grad_out[0])

target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# ===============================
# 3. 预处理
# ===============================
transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# 4. Grad-CAM + 缺陷标注
# ===============================
def detect(image: Image.Image):
    features.clear()
    grads.clear()

    img_tensor = transform(image).unsqueeze(0).to(device)

    start = time.time()
    out = model(img_tensor)
    prob = F.softmax(out, dim=1)
    cls_id = prob.argmax().item()
    confidence = prob[0, cls_id].item()
    out[0, cls_id].backward()
    elapsed = (time.time() - start) * 1000

    fmap = features[0]
    grad = grads[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * fmap).sum(dim=1)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam = cv2.resize(cam, image.size)

    img_np = np.array(image.convert("RGB"))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam_img = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    # 缺陷标注（阈值 + 最大连通域）
    mask = cam > 0.6
    mask = np.uint8(mask * 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ann = img_np.copy()
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(ann, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return (
        CLASSES[cls_id],
        confidence,
        elapsed,
        cam_img,
        ann
    )

# ===============================
# 5. 左侧控制栏
# ===============================
with st.sidebar:
    st.header("1️⃣ 数据导入")
    file = st.file_uploader("上传检测图片", type=["bmp", "jpg", "png"])

    st.divider()
    st.header("2️⃣ 模型选择")
    st.selectbox("模型", ["ResNet18"], disabled=True)
    st.selectbox("设备", ["GPU" if torch.cuda.is_available() else "CPU"], disabled=True)

    st.divider()
    start_btn = st.button("🟢 开始检测", use_container_width=True)

# ===============================
# 6. 主界面
# ===============================
if file:
    image = Image.open(file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 原始图像 / Grad-CAM")

    with col2:
        st.subheader("📌 缺陷标注结果")

    if start_btn:
        with st.spinner("模型检测中..."):
            cls, conf, t, cam_img, ann_img = detect(image)

        col1.image(cam_img, use_column_width=True)
        col2.image(ann_img, use_column_width=True)

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("缺陷类别", cls)
        m2.metric("置信度", f"{conf*100:.2f}%")
        m3.metric("检测耗时", f"{t:.1f} ms")

else:
    st.info("⬅️ 请先在左侧上传检测图片")

