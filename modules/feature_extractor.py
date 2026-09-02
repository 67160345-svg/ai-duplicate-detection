from pathlib import Path
from typing import Dict, List, Tuple
import os
import cv2
import imagehash
import numpy as np
import open_clip
from PIL import Image
import torch
from ultralytics import YOLO
 
# ----------------------------------------------------
# 1. Model Initialization (Singleton Pattern)
# ----------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
# Default: ผูกกับตำแหน่งไฟล์นี้ (สมมติว่า .pt อยู่โฟลเดอร์เดียวกับ modules/)
# ถ้าโครง repo จริงเก็บ weight ไว้คนละที่ ตั้ง env var YOLO_SEG_MODEL_PATH
# ชี้ไป path จริงได้โดยไม่ต้องแก้โค้ด
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "yolov8n-seg.pt"
_MODEL_PATH = os.getenv("YOLO_SEG_MODEL_PATH", str(_DEFAULT_MODEL_PATH))
YOLO_SEG_CONFIDENCE = float(os.getenv("YOLO_SEG_CONFIDENCE", "0.25"))
YOLO_SEG_IOU = float(os.getenv("YOLO_SEG_IOU", "0.7"))
 
print(f"[Tech 1] Loading YOLOv8-Seg on {DEVICE}...")
yolo_model = YOLO(str(_MODEL_PATH))
 
print(f"[Tech 1] Loading OpenCLIP (ViT-B-32) on {DEVICE}...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.to(DEVICE).eval()


def _predict_segmentation(img_rgb: np.ndarray):
  """Run the configured YOLO segmentation model for one image."""
  return yolo_model.predict(
      source=img_rgb,
      conf=YOLO_SEG_CONFIDENCE,
      iou=YOLO_SEG_IOU,
      verbose=False,
  )[0]
 
 
def extract_phash(img_rgb: np.ndarray) -> str:
  """คำนวณ pHash (Perceptual Hash) 64-bit คืนค่าเป็น Hex String 16 หลัก"""
  pil_img = Image.fromarray(img_rgb)
  hash_obj = imagehash.phash(pil_img)
  return str(hash_obj)
 
 
def segment_foreground_background(
    img_rgb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
  """ใช้ YOLOv8-Seg ตัดแยก Foreground (ตัวสินค้า) และ Background
 
  Returns:
      fg_img: ภาพเฉพาะตัวสินค้า (พื้นที่ฉากหลังเป็นสีดำ)
      bg_img: ภาพเฉพาะฉากหลัง (พื้นที่สินค้าเป็นสีดำ)
      has_mask: True ถ้าตรวจพบวัตถุ, False ถ้าไม่พบวัตถุ (Fallback)
  """
  h, w, _ = img_rgb.shape
  result = _predict_segmentation(img_rgb)
 
  # Edge Case: หาก YOLO ตรวจไม่พบ Object ใดๆ เลย ให้ Fallback ใช้ภาพเต็ม
  if result.masks is None or len(result.masks.data) == 0:
    return img_rgb.copy(), img_rgb.copy(), False
 
  # รวม Mask ของทุก Object ที่ตรวจพบในภาพ
  masks_data = result.masks.data.cpu().numpy()
  combined_mask = np.any(masks_data, axis=0).astype(np.uint8)
 
  # Resize Mask ให้ตรงกับขนาดภาพเดิม
  combined_mask_resized = cv2.resize(
      combined_mask, (w, h), interpolation=cv2.INTER_NEAREST
  )
 
  # Bitwise Masking
  fg_img = cv2.bitwise_and(
      img_rgb, img_rgb, mask=combined_mask_resized
  )  # เฉพาะตัวสินค้า
  bg_mask = (1 - combined_mask_resized).astype(np.uint8)
  bg_img = cv2.bitwise_and(img_rgb, img_rgb, mask=bg_mask)  # เฉพาะฉากหลัง
 
  return fg_img, bg_img, True
 
 
def extract_clip_embedding(img_rgb: np.ndarray) -> List[float]:
  """แปลงภาพเป็น CLIP Embedding Vector ขนาด 512 มิติ พร้อมทำ L2 Normalization"""
  pil_img = Image.fromarray(img_rgb)
  image_tensor = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
 
  with torch.no_grad():
    embedding = clip_model.encode_image(image_tensor)
    # L2 Normalization (ทำให้ Unit Vector = 1.0)
    norm_embedding = embedding / embedding.norm(dim=-1, keepdim=True)
 
  vector_list = norm_embedding.cpu().numpy()[0].tolist()
  return [round(float(val), 6) for val in vector_list]
 
 
def extract_all_features(img_rgb: np.ndarray) -> Dict:
  """Main Interface Function สำหรับให้ main.py เรียกใช้งาน
 
  Returns Dict:
      - phash: Hex String
      - fg_vector: List[float] (512 มิติ, L2 Normalized)
      - bg_vector: List[float] (512 มิติ, L2 Normalized)
      - has_segmentation: bool
  """
  # 1. pHash
  phash_val = extract_phash(img_rgb)
 
  # 2. Segmentation
  fg_img, bg_img, has_mask = segment_foreground_background(img_rgb)
 
  # 3. Vector Embeddings
  fg_vector = extract_clip_embedding(fg_img)
  bg_vector = (
      extract_clip_embedding(bg_img) if has_mask else fg_vector.copy()
  )
 
  return {
      "phash": phash_val,
      "fg_vector": fg_vector,
      "bg_vector": bg_vector,
      "has_segmentation": has_mask,
  }