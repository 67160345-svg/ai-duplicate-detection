from typing import Dict
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


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """แปลง Raw File Bytes จาก Form-Data ให้เป็น RGB NumPy Array"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Cannot decode image from uploaded file bytes")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)