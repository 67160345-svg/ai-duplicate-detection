import io
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - installed in the service environment
    register_heif_opener = None

if register_heif_opener:
    register_heif_opener()


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
    """แปลง bytes ของภาพมาตรฐานหรือ HEIC/HEIF เป็น RGB NumPy Array"""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (Image.DecompressionBombError, Image.UnidentifiedImageError, OSError) as exc:
        raise ValueError("Cannot decode image from uploaded file bytes") from exc