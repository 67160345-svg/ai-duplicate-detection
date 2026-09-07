from typing import Dict, List, Tuple
import cv2
import numpy as np


def calculate_quality_scores(img_rgb: np.ndarray) -> Dict[str, float]:
  """คำนวณค่าคุณภาพภาพสำหรับใส่ลง Response Schema (สเกล 0.00 - 1.00)"""
  gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

  # Blur (Laplacian Variance)
  laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
  blur_score = float(np.clip(laplacian_var / 500.0, 0.0, 1.0))

  # Brightness & Contrast
  brightness_score = float(np.mean(gray) / 255.0)
  contrast_score = float(np.clip(np.std(gray) / 128.0, 0.0, 1.0))

  # Noise (ประมาณการจากค่าความแปรปรวนหลัง Median Filter)
  noise_sigma = np.std(
      gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
  )
  noise_score = float(np.clip(noise_sigma / 40.0, 0.0, 1.0))

  return {
      'blur': round(blur_score, 4),
      'brightness': round(brightness_score, 4),
      'contrast': round(contrast_score, 4),
      'noise': round(noise_score, 4),
  }


def check_image_quality(img_rgb: np.ndarray) -> Tuple[bool, List[str]]:
    """คืนค่า True ถ้า image มีคุณภาพเพียงพอสำหรับ detection, False พร้อมเหตุผล"""
    if img_rgb is None or img_rgb.size == 0:
        return False, ["ภาพไม่ถูกต้องหรือว่างเปล่า"]

    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    quality_scores = calculate_quality_scores(img_rgb)

    issues: List[str] = []

    # Resolution guard: images too small are unreliable for segmentation
    if max(h, w) < 224 or min(h, w) < 128:
        issues.append("ความละเอียดต่ำเกินไป")

    # Basic quality gate: reject only clearly unusable images.
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 30:
        issues.append("ภาพเบลอ")

    brightness = float(np.mean(gray))
    if brightness < 30:
        issues.append("แสงมืดเกินไป")

    if quality_scores["contrast"] < 0.06:
        issues.append("ความคม contrast ต่ำเกินไป")

    # Keep the rule intentionally simple to avoid rejecting normal product shots.
    # Off-center detection is omitted here because it is too noisy for a basic gate.

    # Multiple weak signals should fail conservatively
    if not issues:
        return True, []

    # Keep output concise and readable for API consumers.
    deduped = []
    seen = set()
    for issue in issues:
        if issue not in seen:
            deduped.append(issue)
            seen.add(issue)
    return False, deduped


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """แปลง Raw File Bytes จาก Form-Data ให้เป็น RGB NumPy Array"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Cannot decode image from uploaded file bytes")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)