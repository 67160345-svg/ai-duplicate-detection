import json
from typing import Dict, List, Tuple, Optional
import imagehash
import numpy as np

DECISION_DUPLICATE = "DUPLICATE"
DECISION_REVIEW = "REVIEW"
DECISION_UNIQUE = "UNIQUE"
DECISION_INVALID_DATA = "INVALID_DATA"
DECISION_SPAM = "SPAM"

DEFAULT_SIMILARITY_THRESHOLDS = {
    "exact_duplicate": 0.99,
    "near_duplicate": 0.95,
    "possible_duplicate": 0.85,
    "review": 0.70,
}


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    try:
        if isinstance(vec1, str):
            vec1 = json.loads(vec1)
        if isinstance(vec2, str):
            vec2 = json.loads(vec2)
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    if v1.ndim != 1 or v2.ndim != 1 or v1.shape != v2.shape:
        return 0.0
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    sim = float(np.dot(v1, v2) / (norm1 * norm2))
    return float(np.clip(sim, 0.0, 1.0))

def compute_phash_similarity(phash1: str, phash2: str) -> Tuple[int, float]:
    h1 = imagehash.hex_to_hash(phash1)
    h2 = imagehash.hex_to_hash(phash2)
    dist = h1 - h2
    sim_score = max(0.0, 1.0 - (dist / 64.0))
    return dist, round(sim_score, 4)

def _has_valid_features(
    fg_vector: List[float], bg_vector: List[float], phash: str
) -> bool:
    """Reject malformed feature records before they enter business rules."""
    try:
        if not fg_vector or not bg_vector or not phash:
            return False
        if len(fg_vector) != len(bg_vector):
            return False
        if not np.all(np.isfinite(np.asarray(fg_vector, dtype=np.float32))):
            return False
        if not np.all(np.isfinite(np.asarray(bg_vector, dtype=np.float32))):
            return False
        imagehash.hex_to_hash(phash)
    except (TypeError, ValueError):
        return False
    return True


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "yes", "1"}


def classify_dataset_record(
    record: Dict,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """Classify a numeric dataset row using the Phase 0-1 business rules.

    Dataset rows contain precomputed signals, so this adapter lets evaluation
    reuse the same thresholds without pretending to run image inference.
    """
    active_thresholds = {**DEFAULT_SIMILARITY_THRESHOLDS, **(thresholds or {})}
    similarity_value = record.get("Embedding_Similarity", record.get("Similarity_Score"))
    try:
        similarity = float(similarity_value)
        if not np.isfinite(similarity) or not 0.0 <= similarity <= 1.0:
            return DECISION_INVALID_DATA
    except (TypeError, ValueError):
        return DECISION_INVALID_DATA

    perceptual_hash = record.get("Perceptual_Hash")
    if perceptual_hash not in (None, ""):
        try:
            imagehash.hex_to_hash(str(perceptual_hash))
        except (TypeError, ValueError):
            return DECISION_INVALID_DATA

    business_rule = str(record.get("Business_Rule", "")).upper()
    if business_rule == "BR010":
        return DECISION_INVALID_DATA if not _is_true(record.get("Seller_Match", True)) else DECISION_SPAM
    if business_rule in {"BR011", "BR012"}:
        return DECISION_UNIQUE
    if any(
        _is_true(record.get(field))
        for field in (
            "Watermark_Detected",
            "Screenshot_Detected",
            "AI_Image_Detected",
            "Stock_Image_Detected",
        )
    ):
        return DECISION_REVIEW
    if similarity >= active_thresholds["near_duplicate"]:
        return DECISION_DUPLICATE
    if similarity >= active_thresholds["possible_duplicate"]:
        return DECISION_REVIEW
    if similarity >= active_thresholds["review"]:
        return DECISION_REVIEW
    return DECISION_UNIQUE


def evaluate_baseline_decision(
    new_fg_vec: List[float],
    new_bg_vec: List[float],
    new_phash: str,
    db_records: List[Dict],
    fg_threshold: float = 0.85,
    phash_distance_threshold: int = 5,
    review_threshold: float = 0.75,
) -> Tuple[str, float, float, float, str, Optional[Dict]]:
    """คืนค่า decision, scores, reason และ record ที่ match ที่สุด.

    Phase 0-1 rules: invalid input, exact/near duplicate, review band และ unique.
    """
    if not _has_valid_features(new_fg_vec, new_bg_vec, new_phash):
        return (
            DECISION_INVALID_DATA,
            0.0,
            0.0,
            0.0,
            "ข้อมูล feature ของภาพไม่ถูกต้องหรือไม่ครบถ้วน",
            None,
        )

    if not db_records:
        return (
            DECISION_UNIQUE,
            0.0,
            0.0,
            0.0,
            "ระบบยังไม่มีข้อมูลภาพอ้างอิงในระบบ (First Image Entry)",
            None,
        )
 
    # เก็บ "ผู้ท้าชิง" ของแต่ละ metric แยกกันคนละตัวแปร ห้ามปนกัน
    # เพราะ record ที่ชนะ phash อาจไม่ใช่ record เดียวกับที่ชนะ fg_sim
    best_fg_sim = 0.0
    best_fg_bg_sim = 0.0
    best_fg_record = db_records[0]
 
    min_phash_dist = 64
    best_phash_sim = 0.0
    best_phash_record = None
 
    for record in db_records:
        fg_sim = compute_cosine_similarity(new_fg_vec, record["fg_vector"])
        bg_sim = compute_cosine_similarity(new_bg_vec, record.get("bg_vector", new_bg_vec))
        dist, p_sim = compute_phash_similarity(new_phash, record["phash"])
 
        if fg_sim > best_fg_sim:
            best_fg_sim = fg_sim
            best_fg_bg_sim = bg_sim
            best_fg_record = record
 
        if dist < min_phash_dist:
            min_phash_dist = dist
            best_phash_sim = p_sim
            best_phash_record = record
 
    def _record_id(record: Dict) -> str:
        return record.get("product_id", record.get("image_url", "Unknown"))
 
    # เลือก matched_record ตาม "เกณฑ์ที่ใช้ตัดสินใจจริง" ในแต่ละ branch
    # แทนที่จะใช้ record ตัวเดียวปนกันทุกกรณีเหมือนเดิม
    if min_phash_dist <= phash_distance_threshold:
        decision = DECISION_DUPLICATE
        matched_record = best_phash_record
        matched_id = _record_id(matched_record)
        scenario = "EXACT_DUPLICATE" if min_phash_dist == 0 else "NEAR_EXACT_DUPLICATE"
        reason = (
            f"ตรวจพบภาพซ้ำประเภท {scenario} ตรงกับสินค้า [{matched_id}] "
            f"(pHash Distance: {min_phash_dist}/64, Foreground Visual Match: {best_fg_sim*100:.2f}%) "
            f"โครงสร้างพิกเซลและเลย์เอาต์หลักตรงกันเกือบสมบูรณ์"
        )
    elif best_fg_sim >= fg_threshold:
        decision = DECISION_DUPLICATE
        matched_record = best_fg_record
        matched_id = _record_id(matched_record)
        reason = (
            f"ตรวจพบสินค้าชิ้นเดียวกัน (Physical Duplicate) ตรงกับสินค้า [{matched_id}] "
            f"โดยตัวสินค้า (Foreground) มีความเหมือน {best_fg_sim*100:.2f}% (เกินเกณฑ์ขั้นต่ำ {fg_threshold*100:.0f}%) "
            f"แม้ฉากหลังอาจแตกต่างกัน (Background Match: {best_fg_bg_sim*100:.2f}%)"
        )
    elif best_fg_sim >= review_threshold:
        decision = DECISION_REVIEW
        matched_record = best_fg_record
        matched_id = _record_id(matched_record)
        reason = (
            f"พบภาพที่มีความคล้ายสูงและต้องตรวจสอบต่อ ตรงกับสินค้า [{matched_id}] "
            f"โดยตัวสินค้ามีความเหมือน {best_fg_sim*100:.2f}% "
            f"(ช่วง Review: {review_threshold*100:.0f}-{fg_threshold*100:.0f}%)"
        )
    else:
        decision = DECISION_UNIQUE
        # ไม่ซ้ำ แต่ยังรายงาน record ที่ใกล้เคียงที่สุด (ตาม fg_sim) ไว้เพื่อ reference
        matched_record = best_fg_record
        matched_id = _record_id(matched_record)
        reason = (
            f"สินค้ามีเอกลักษณ์เฉพาะตัว (Unique) เทียบกับสินค้าใกล้เคียงที่สุด [{matched_id}] "
            f"มีคะแนนความเหมือนตัวสินค้าเพียง {best_fg_sim*100:.2f}% (ไม่ถึงเกณฑ์ {fg_threshold*100:.0f}%)"
        )
 
    return decision, round(best_fg_sim, 4), round(best_fg_bg_sim, 4), best_phash_sim, reason, matched_record