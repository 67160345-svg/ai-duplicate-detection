"""Module: scoring_engine.py

Author: Tech 2 / Tech Lead
Description: Cosine Distance calculation and Phase 1 Baseline Decision Engine.
"""

from typing import Dict, List, Tuple
import imagehash
import numpy as np


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
  """คำนวณ Cosine Similarity ระหว่าง 2 Normalized Vectors (คืนค่า 0.00 - 1.00)"""
  v1, v2 = np.array(vec1, dtype=np.float32), np.array(vec2, dtype=np.float32)
  norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
  if norm1 == 0.0 or norm2 == 0.0:
    return 0.0
  sim = float(np.dot(v1, v2) / (norm1 * norm2))
  return float(np.clip(sim, 0.0, 1.0))


def compute_phash_similarity(phash1: str, phash2: str) -> Tuple[int, float]:
  """คำนวณ Hamming Distance ของ pHash และแปลงเป็น Similarity Score (0.00 - 1.00)

  Hamming Distance: 0 = เหมือนเป๊ะ 100%, > 10 = คนละรูป
  """
  h1 = imagehash.hex_to_hash(phash1)
  h2 = imagehash.hex_to_hash(phash2)
  dist = h1 - h2  # Hamming distance (0 ถึง 64)
  sim_score = max(0.0, 1.0 - (dist / 64.0))
  return dist, round(sim_score, 4)


def evaluate_baseline_decision(
    new_fg_vec: List[float],
    new_bg_vec: List[float],
    new_phash: str,
    db_records: List[Dict],
    fg_threshold: float = 0.85,
    phash_distance_threshold: int = 5,
) -> Tuple[str, float, float, float, str]:
  """เปรียบเทียบภาพใหม่กับฐานข้อมูลภาพเดิม และตัดสินผล Baseline

  Rules:
  1. Exact Duplicate: pHash Difference <= 5 (หรือ pHash Match >= 0.92)
  2. Physical Duplicate: Foreground Cosine Sim >= 0.85
  """
  # กรณีเป็นรูปแรกสุดของระบบ
  if not db_records:
    return "UNIQUE", 0.0, 0.0, 0.0, "First image in reference database"

  max_fg_sim = 0.0
  max_bg_sim = 0.0
  min_phash_dist = 64
  best_phash_sim = 0.0

  for record in db_records:
    # 1. Cosine Sim
    fg_sim = compute_cosine_similarity(new_fg_vec, record["fg_vector"])
    bg_sim = compute_cosine_similarity(
        new_bg_vec, record.get("bg_vector", new_bg_vec)
    )

    max_fg_sim = max(max_fg_sim, fg_sim)
    max_bg_sim = max(max_bg_sim, bg_sim)

    # 2. pHash Distance
    dist, p_sim = compute_phash_similarity(new_phash, record["phash"])
    if dist < min_phash_dist:
      min_phash_dist = dist
      best_phash_sim = p_sim

  # Decision Rules
  if min_phash_dist <= phash_distance_threshold:
    decision = "DUPLICATE"
    reason = (
        f"Exact Hash Match (pHash Diff: {min_phash_dist},"
        f" FG-Sim: {max_fg_sim:.2f})"
    )
  elif max_fg_sim >= fg_threshold:
    decision = "DUPLICATE"
    reason = (
        f"High Foreground Visual Similarity (FG-Sim: {max_fg_sim:.2f} >="
        f" {fg_threshold})"
    )
  else:
    decision = "UNIQUE"
    reason = (
        f"Unique Item (Max FG-Sim: {max_fg_sim:.2f} < Threshold {fg_threshold})"
    )

  return (
      decision,
      round(max_fg_sim, 4),
      round(max_bg_sim, 4),
      best_phash_sim,
      reason,
  )