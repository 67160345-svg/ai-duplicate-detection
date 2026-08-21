from typing import Dict, Tuple
import cv2
import numpy as np
import requests


def load_image_from_url_or_path(image_source: str) -> np.ndarray:
  """โหลดรูปภาพจาก URL หรือ Local Path และแปลงเป็น RGB NumPy Array (uint8)"""
  headers = {
      'User-Agent': (
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,'
          ' like Gecko) Chrome/120.0.0.0 Safari/537.36'
      )
  }

  if image_source.startswith(('http://', 'https://')):
    resp = requests.get(image_source, headers=headers, timeout=12)
    resp.raise_for_status()
    image_bytes = np.frombuffer(resp.content, np.uint8)
    img_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
  else:
    img_bgr = cv2.imread(image_source, cv2.IMREAD_COLOR)

  if img_bgr is None:
    raise ValueError(f'Cannot decode image from source: {image_source}')

  # แปลงเป็น RGB เสมอ
  return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def preprocess_image(
    img_rgb: np.ndarray, target_size: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, np.ndarray]:
  """1. Resize 2. Denoise (Bilateral Filter เพื่อความเร็ว) 3. Min-Max Normalization

  Returns:
      denoised_img: ภาพ uint8 (0-255) สำหรับส่งเข้า YOLOv8 / Visualizers
      norm_img_model: ภาพ float32 (0.0-1.0) สำหรับส่งเข้า ML Model
  """
  # 1. Resize
  resized_img = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)

  # 2. Denoise แบบ Fast (รักษาขอบภาพ)
  denoised_img = cv2.bilateralFilter(
      resized_img, d=5, sigmaColor=35, sigmaSpace=35
  )

  # 3. Min-Max Normalization (0.0 - 1.0)
  norm_img_model = denoised_img.astype(np.float32) / 255.0

  return denoised_img, norm_img_model


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