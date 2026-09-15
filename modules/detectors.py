from __future__ import annotations

from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


def _to_gray(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def detect_screenshot(img_rgb: np.ndarray) -> Tuple[bool, str]:
    """Simple screenshot/UI chrome detector for Phase 3 / BR006.
    This is intentionally conservative: flag only when strong UI-like patterns appear.
    """
    gray = _to_gray(img_rgb)
    h, w = gray.shape

    top_band = gray[: max(24, h // 8), :]
    bottom_band = gray[-max(24, h // 8):, :]
    left_band = gray[:, : max(24, w // 10)]
    right_band = gray[:, -max(24, w // 10):]

    top_std = float(np.std(top_band))
    bottom_std = float(np.std(bottom_band))
    left_std = float(np.std(left_band))
    right_std = float(np.std(right_band))

    edge_map = cv2.Canny(gray, 80, 180)
    border_bands = [
        edge_map[: max(24, h // 12), :],
        edge_map[-max(24, h // 12):, :],
        edge_map[:, : max(24, w // 12)],
        edge_map[:, -max(24, w // 12):],
    ]
    border_edge_density = max(
        float(np.count_nonzero(band) / max(1, band.size))
        for band in border_bands
    )
    band_variance = max(top_std, bottom_std, left_std, right_std)
    chrome_signal = band_variance > 55 and border_edge_density > 0.025

    # OCR check: common status labels in screenshots
    ocr_signal = False
    if pytesseract is not None:
        try:
            pil = Image.fromarray(img_rgb)
            text = pytesseract.image_to_string(pil, config="--psm 6")
            screen_tokens = [
                "wifi", "battery", "signal", "time", "status",
                "notification", "search", "home", "menu"
            ]
            lowered = text.lower()
            ocr_signal = any(token in lowered for token in screen_tokens)
        except Exception:
            pass

    if chrome_signal and ocr_signal:
        return True, "พบ UI chrome และข้อความ status bar ที่เข้าลักษณะ screenshot"
    if chrome_signal:
        return True, "พบเส้นขอบ UI ที่เด่นผิดปกติบริเวณขอบภาพ"
    if ocr_signal:
        return True, "พบข้อความที่เข้าลักษณะ UI status bar / screenshot"

    return False, "ไม่พบลักษณะ screenshot บริเวณ UI chrome"


def detect_watermark(img_rgb: np.ndarray) -> Tuple[bool, str]:
    """Watermark detection for BR007. Conservative: flag only obvious overlays."""
    gray = _to_gray(img_rgb)
    h, w = gray.shape

    # Look for text-like components concentrated in a corner or repeated over the image.
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8)

    large_regions = []
    text_like_regions = []
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if area < 150:
            continue
        if ww / max(1, w) > 0.55 or hh / max(1, h) > 0.55:
            large_regions.append((x, y, ww, hh, area))
        if 0.01 * w < ww < 0.45 * w and 0.01 * h < hh < 0.15 * h:
            text_like_regions.append((x, y, ww, hh, area))

    def is_corner(x: int, y: int, ww: int, hh: int) -> bool:
        corner_width = 0.3 * w
        corner_height = 0.3 * h
        return (
            (x < corner_width and y < corner_height)
            or (x + ww > w - corner_width and y < corner_height)
            or (x < corner_width and y + hh > h - corner_height)
            or (x + ww > w - corner_width and y + hh > h - corner_height)
        )

    coverage_signal = False
    if large_regions:
        x, y, ww, hh, area = max(large_regions, key=lambda item: item[4])
        if ww * hh > 0.08 * h * w:
            coverage_signal = True

    corner_ocr_signal = False
    if pytesseract is not None:
        try:
            pil = Image.fromarray(img_rgb)
            data = pytesseract.image_to_data(pil, config="--psm 11", output_type=pytesseract.Output.DICT)
            trusted_words = []
            corner_words = []
            for index, text in enumerate(data.get("text", [])):
                if not text.strip() or float(data["conf"][index]) < 55:
                    continue
                trusted_words.append(index)
                if is_corner(
                    int(data["left"][index]),
                    int(data["top"][index]),
                    int(data["width"][index]),
                    int(data["height"][index]),
                ):
                    corner_words.append(index)
            corner_ocr_signal = bool(corner_words)
        except Exception:
            pass

    corner_text_regions = [
        region for region in text_like_regions if is_corner(*region[:4])
    ]
    repeated_corner_text_signal = len(corner_text_regions) >= 3
    if coverage_signal and (corner_ocr_signal or repeated_corner_text_signal):
        return True, "พบ overlay ที่มีพื้นที่ปกคลุมและลักษณะข้อความ/โลโก้"
    if repeated_corner_text_signal and corner_ocr_signal:
        return True, "พบข้อความหรือโลโก้ซ้ำบนพื้นที่ภาพที่เข้าลักษณะ watermark"

    return False, "ไม่พบ watermark / logo overlay ที่ชัดเจน"


def detect_ai_generated(img_rgb: np.ndarray) -> Tuple[bool, str]:
    """Very lightweight AI-artifact detector. BR008.
    This is intentionally soft and conservative: only report strong artifact patterns.
    """
    gray = _to_gray(img_rgb)
    h, w = gray.shape

    # Uniform images are not AI artifacts; they are simply plain or low-information shots.
    gray_std = float(np.std(gray))
    if gray_std < 8:
        return False, "ภาพมีพื้นผิวเรียบ/ไม่มีข้อมูลเพียงพอ ไม่ใช่ artifact ของภาพ AI"

    # Strong noise or unnatural texture patterns are common AI-generated artifacts.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.absdiff(gray, blur)
    noise_ratio = float(np.mean(noise) / 255.0)

    # Unnatural texture often shows high local randomness and odd edge continuity.
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = float(np.count_nonzero(edges) / max(1, h * w))

    noise_signal = noise_ratio > 0.16 and edge_ratio > 0.15

    # Good AI-generated images often have over-smooth edges or weird deformations,
    # but only when the image also has enough structure to justify the check.
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    smooth_signal = gray_std > 12 and laplacian_var < 18 and edge_ratio < 0.04
    if noise_signal:
        return True, "พบ texture noise และ edge density สูงผิดปกติที่เข้าลักษณะ AI artifact"
    if smooth_signal:
        return True, "พบพื้นผิวเรียบเกินไปพร้อม edge continuity ต่ำที่เข้าลักษณะ AI artifact"

    return False, "ไม่พบ artifact ของภาพ AI ที่ชัดเจน"


def run_detectors(img_rgb: np.ndarray) -> Dict[str, Dict[str, object]]:
    """Run all Phase 3 detectors and return structured output for API use."""
    results = {}
    for name, fn in {
        "screenshot": detect_screenshot,
        "watermark": detect_watermark,
        "ai_generated": detect_ai_generated,
    }.items():
        flagged, reason = fn(img_rgb)
        results[name] = {
            "flagged": flagged,
            "reason": reason,
        }
    return results
