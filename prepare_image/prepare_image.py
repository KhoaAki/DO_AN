import cv2
import numpy as np
import tempfile
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2
from docx import Document
from docx.shared import Pt
from .model import load_model

def process_image(path, resize_enabled=True, target_height=720):
    # === STEP 1: Load ảnh gốc và xoay ảnh cho thẳng ===
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.rad2deg(theta)
            if 80 < angle < 100:
                angles.append(angle - 90)
    if angles:
        avg_angle = np.mean(angles)
        (h, w) = img_bgr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
        rotated = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        print(f"[INFO] Ảnh đã được xoay thẳng với góc: {avg_angle:.2f}°")
    else:
        print("[WARN] Không phát hiện góc xoay → giữ nguyên ảnh gốc")
        rotated = img_rgb.copy()

    H, W = rotated.shape[:2]
    # === STEP 2: Dùng DocTR để detect text box ===
    doc = DocumentFile.from_images(path)
    model = ocr_predictor(det_arch="db_resnet50", pretrained=True)
    result = model(doc)
    # Nếu ảnh có alpha (4 kênh), chuyển về RGB
    if rotated.shape[2] == 4:
        rotated = cv2.cvtColor(rotated, cv2.COLOR_RGBA2RGB)
    # === STEP 3: Ghi ảnh xoay ra tạm để dùng với DocTR ===
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))

    doc_rotated = DocumentFile.from_images(temp_path)
    result_rotated = model(doc_rotated)
    # === STEP 4: Tìm bounding box chứa toàn bộ văn bản ===
    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    for block in result_rotated.pages[0].blocks:
        for line in block.lines:
            (x_min, y_min), (x_max, y_max) = line.geometry
            xmins.append(int(x_min * W))
            ymins.append(int(y_min * H))
            xmaxs.append(int(x_max * W))
            ymaxs.append(int(y_max * H))

    if not xmins or not ymins:
        raise ValueError("❌ Không phát hiện được văn bản trong ảnh đã xoay.")
    x1, y1 = max(0, min(xmins)), max(0, min(ymins))
    x2, y2 = min(W, max(xmaxs)), min(H, max(ymaxs))
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)
    # === STEP 5: Cắt vùng văn bản từ ảnh đã xoay ===
    cropped = rotated[y1:y2, x1:x2]
    # === STEP 6: Resize nếu cần ===
    text_height = y2 - y1
    text_width = x2 - x1
    if resize_enabled and text_height > 0:
        scale = target_height / text_height
        new_w = int(text_width * scale)
        final_image = cv2.resize(cropped, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
    else:
        final_image = cropped

    return final_image  # Ảnh gốc, ảnh đã xoay, ảnh cắt và resize


def restore_image(final_image, device='cpu', checkpoint_path= "users/main_function/prepare_image/checkpoint.pth"):
    # Load model
    model = load_model(checkpoint_path, device=device)
    # Chuyển sang grayscale nếu ảnh là RGB
    if final_image.ndim == 3:
        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2GRAY)
    # Chuẩn hóa ảnh về [0, 1]
    img = final_image.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    # Khôi phục ảnh
    with torch.no_grad():
        output_tensor = model(input_tensor)
        output_tensor = torch.clamp(output_tensor, 0., 1.)
    # Chuyển về ảnh numpy uint8 (0–255)
    restored_img = (output_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    return restored_img
