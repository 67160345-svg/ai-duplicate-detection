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

# Phase 2: supported product detection focus.
# These are not a hard training class list for YOLO, but they define which product
# families are considered valid for the immediate MVP: phones, tablets, cameras,
# computing accessories, apparel, tools, beauty, and health items.
SUPPORTED_PRODUCT_HINTS = {
    "cell phone", "laptop", "keyboard", "mouse", "tv", "monitor",
    "tablet", "camera", "remote", "book", "bottle", "suitcase",
    "backpack", "scissors", "toothbrush", "perfume", "hair dryer",
    "watch", "shoes", "t-shirt", "shirt", "dress", "jeans",
    "bag", "headphones", "microphone", "router", "speaker",
}
 
# Default: ผูกกับตำแหน่งไฟล์นี้ (สมมติว่า .pt อยู่โฟลเดอร์เดียวกับ modules/)
# ถ้าโครง repo จริงเก็บ weight ไว้คนละที่ ตั้ง env var YOLO_SEG_MODEL_PATH
# ชี้ไป path จริงได้โดยไม่ต้องแก้โค้ด
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "yolov8n-seg.pt"
_MODEL_PATH = os.getenv("YOLO_SEG_MODEL_PATH", str(_DEFAULT_MODEL_PATH))
YOLO_SEG_CONFIDENCE = float(os.getenv("YOLO_SEG_CONFIDENCE", "0.25"))
YOLO_SEG_IOU = float(os.getenv("YOLO_SEG_IOU", "0.7"))
YOLO_SEG_MASK_CONFIDENCE = float(os.getenv("YOLO_SEG_MASK_CONFIDENCE", "0.35"))
YOLO_SEG_MAX_OBJECTS = int(os.getenv("YOLO_SEG_MAX_OBJECTS", "3"))
 
print(f"[Tech 1] Loading YOLOv8-Seg on {DEVICE}...")
yolo_model = YOLO(str(_MODEL_PATH))
 
print(f"[Tech 1] Loading OpenCLIP (ViT-B-32) on {DEVICE}...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.to(DEVICE).eval()


def _get_detection_labels(result) -> List[str]:
    """Try to recover class names from YOLO result when available."""
    labels: List[str] = []
    if result is None:
        return labels
    try:
        if hasattr(result, "boxes") and result.boxes is not None:
            cls_ids = result.boxes.cls.cpu().tolist()
            for cls_id in cls_ids:
                idx = int(cls_id)
                label = yolo_model.model.names.get(idx, str(idx)).lower()
                labels.append(label)
    except Exception:
        return labels
    return labels


def _evaluate_segmentation_support(result, h: int, w: int) -> Tuple[bool, str]:
    """Conservative gate: only allow segmentation when the object appears viable for product detection."""
    if result is None or result.masks is None or len(result.masks.data) == 0:
        return False, "ไม่พบวัตถุที่สามารถแยกสินค้าออกจากพื้นหลังได้"

    masks_data = result.masks.data.cpu().numpy()
    if len(masks_data) > 4:
        return False, "ภาพมีวัตถุมากเกินไปสำหรับการตรวจจับสินค้ารูปแบบนี้"

    area_ratios = []
    for mask in masks_data:
        mask_u8 = mask.astype(np.uint8)
        area_ratio = float(np.count_nonzero(mask_u8) / max(1, h * w))
        area_ratios.append(area_ratio)

    if any(ratio < 0.03 for ratio in area_ratios):
        return False, "วัตถุมีขนาดเล็กเกินไปสำหรับการ detect ที่เชื่อถือได้"

    if any(ratio > 0.9 for ratio in area_ratios):
        return False, "วัตถุครอบคลุมภาพเกินไป จึงไม่เหมาะกับการแยก foreground/background"

    labels = _get_detection_labels(result)
    if labels:
        lower_labels = {label.lower() for label in labels}
        if lower_labels.issubset({"person", "animal"}):
            return False, "ตรวจพบคน/สัตว์เป็นวัตถุหลัก ไม่ใช่สินค้าอีคอมเมิร์ซที่สามารถ detect ได้"

        if not any(label in SUPPORTED_PRODUCT_HINTS for label in lower_labels):
            # This is intentionally conservative: only proceed when the detection is clearly on a product-like object.
            # We still allow ambiguous cases through if they look like a product silhouette, but not when they are
            # obviously outside the targeted category families for this phase.
            return False, "วัตถุที่ตรวจพบไม่อยู่ใน focus category ของ phase 2"

    return True, "วัตถุพร้อมสำหรับการแยก foreground/background"


def _predict_segmentation(img_rgb: np.ndarray):
    """Run the configured YOLO segmentation model for one image."""
    return yolo_model.predict(
        source=img_rgb,
        conf=YOLO_SEG_CONFIDENCE,
        iou=YOLO_SEG_IOU,
        max_det=YOLO_SEG_MAX_OBJECTS,
        verbose=False,
    )[0]


def _build_product_mask(result, h: int, w: int) -> np.ndarray | None:
    """Select viable product masks and clean their edges before compositing."""
    if result is None or result.masks is None or len(result.masks.data) == 0:
        return None

    masks_data = result.masks.data.cpu().numpy()
    confidences = []
    class_ids = []
    if getattr(result, "boxes", None) is not None:
        confidences = result.boxes.conf.cpu().numpy().tolist()
        class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()

    selected_masks = []
    for index, mask in enumerate(masks_data):
        confidence = float(confidences[index]) if index < len(confidences) else 1.0
        if confidence < YOLO_SEG_MASK_CONFIDENCE:
            continue

        mask_resized = cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
        area_ratio = float(np.count_nonzero(mask_resized) / max(1, h * w))
        if area_ratio < 0.03 or area_ratio > 0.90:
            continue

        if class_ids and index < len(class_ids):
            label = yolo_model.model.names.get(class_ids[index], str(class_ids[index])).lower()
            if label not in SUPPORTED_PRODUCT_HINTS:
                continue
        selected_masks.append((confidence, area_ratio, mask_resized))

    if not selected_masks:
        return None

    # Prefer confident, useful-sized objects and avoid tiny secondary detections.
    selected_masks.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected_masks = selected_masks[:YOLO_SEG_MAX_OBJECTS]
    combined_mask = np.any(
        [item[2] > 0 for item in selected_masks], axis=0
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    return (combined_mask > 0).astype(np.uint8)
 
 
def extract_phash(img_rgb: np.ndarray) -> str:
  """คำนวณ pHash (Perceptual Hash) 64-bit คืนค่าเป็น Hex String 16 หลัก"""
  pil_img = Image.fromarray(img_rgb)
  hash_obj = imagehash.phash(pil_img)
  return str(hash_obj)
 
 
def segment_foreground_background(
        img_rgb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """ใช้ YOLOv8-Seg ตัดแยก Foreground (ตัวสินค้า) และ Background."""
    h, w, _ = img_rgb.shape
    result = _predict_segmentation(img_rgb)

    if result.masks is None or len(result.masks.data) == 0:
        return img_rgb.copy(), img_rgb.copy(), False, "ไม่พบวัตถุที่สามารถแยกสินค้าออกจากพื้นหลังได้"

    supported, reason = _evaluate_segmentation_support(result, h, w)
    if not supported:
        return img_rgb.copy(), img_rgb.copy(), False, reason

    combined_mask_resized = _build_product_mask(result, h, w)
    if combined_mask_resized is None:
        return img_rgb.copy(), img_rgb.copy(), False, "ไม่พบ mask สินค้าที่มีความมั่นใจเพียงพอ"

    final_area_ratio = float(np.count_nonzero(combined_mask_resized) / max(1, h * w))
    if final_area_ratio < 0.03 or final_area_ratio > 0.90:
        return img_rgb.copy(), img_rgb.copy(), False, "mask สินค้าไม่อยู่ในขนาดที่เหมาะสมสำหรับการแยกพื้นหลัง"

    fg_img = cv2.bitwise_and(img_rgb, img_rgb, mask=combined_mask_resized)
    bg_mask = (1 - combined_mask_resized).astype(np.uint8)
    bg_img = cv2.bitwise_and(img_rgb, img_rgb, mask=bg_mask)

    return fg_img, bg_img, True, reason
 
 
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
      - segmentation_reason: str
  """
  # 1. pHash
  phash_val = extract_phash(img_rgb)
 
  # 2. Segmentation
  fg_img, bg_img, has_mask, segmentation_reason = segment_foreground_background(img_rgb)
 
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
      "segmentation_reason": segmentation_reason,
  }