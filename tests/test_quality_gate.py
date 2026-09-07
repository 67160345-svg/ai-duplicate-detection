import unittest
import numpy as np

from modules.preprocessor import check_image_quality
from modules.scoring_engine import evaluate_baseline_decision


class QualityGateTests(unittest.TestCase):
    def test_blurry_dark_low_resolution_image_is_reviewed(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        ok, issues = check_image_quality(img)
        self.assertFalse(ok)
        self.assertTrue(any("ภาพเบลอ" in issue or "แสงมืดเกินไป" in issue or "ความละเอียดต่ำเกินไป" in issue for issue in issues))

    def test_clear_well_lit_image_is_accepted(self):
        rng = np.random.default_rng(0)
        base = np.full((800, 800, 3), 220, dtype=np.uint8)
        noise = rng.normal(0, 12, size=base.shape).astype(np.int16)
        img = np.clip(base + noise, 0, 255).astype(np.uint8)
        ok, issues = check_image_quality(img)
        self.assertTrue(ok)
        self.assertEqual(issues, [])

    def test_uniform_image_is_not_marked_as_ai_generated(self):
        from modules.detectors import detect_ai_generated
        img = np.full((400, 400, 3), 200, dtype=np.uint8)
        flagged, reason = detect_ai_generated(img)
        self.assertFalse(flagged)
        self.assertIn("ไม่ใช่ artifact", reason)

    def test_phase1_thresholds_keep_85_percent_match_in_review(self):
        decision, *_ = evaluate_baseline_decision(
            new_fg_vec=[1.0, 0.0],
            new_bg_vec=[1.0, 0.0],
            new_phash="0000000000000000",
            db_records=[
                {
                    "product_id": "reference-1",
                    "fg_vector": [0.85, 0.5267827],
                    "bg_vector": [1.0, 0.0],
                    "phash": "ffffffffffffffff",
                }
            ],
            fg_threshold=0.95,
            review_threshold=0.70,
        )
        self.assertEqual(decision, "REVIEW")

    def test_phase1_thresholds_require_95_percent_match_for_duplicate(self):
        decision, *_ = evaluate_baseline_decision(
            new_fg_vec=[1.0, 0.0],
            new_bg_vec=[1.0, 0.0],
            new_phash="0000000000000000",
            db_records=[
                {
                    "product_id": "reference-1",
                    "fg_vector": [0.96, 0.28],
                    "bg_vector": [1.0, 0.0],
                    "phash": "ffffffffffffffff",
                }
            ],
            fg_threshold=0.95,
            review_threshold=0.70,
        )
        self.assertEqual(decision, "DUPLICATE")


if __name__ == "__main__":
    unittest.main()
