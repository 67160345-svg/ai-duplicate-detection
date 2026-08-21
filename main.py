from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# นำเข้าฟังก์ชันจริงจากทั้ง 3 โมดูล
from modules.feature_extractor import extract_all_features
from modules.preprocessor import calculate_quality_scores, load_image_from_url_or_path
from modules.scoring_engine import evaluate_baseline_decision

app = FastAPI(
    title="Duplicate Image Detection Engine",
    description="Phase 1: Real Baseline Pipeline Integration",
    version="1.0.0",
)

# In-Memory Database สำหรับเก็บข้อมูลภาพที่เคยตรวจในระบบ
DB_REFERENCE_IMAGES: List[Dict] = []


# ==========================================
# 1. กำหนด Pydantic Schema (Request Payload)
# ==========================================
class ProductInfo(BaseModel):
    uploaded_product_id: str
    seller_id: str
    listing_id: str
    uploaded_category: str
    uploaded_brand: str
    uploaded_product_name: str


class DetectionRequest(BaseModel):
    uploaded_images: List[str]
    uploaded_product_info: ProductInfo


# ==========================================
# 2. กำหนด Pydantic Schema (Response Payload)
# ==========================================
class QualityScores(BaseModel):
    blur: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    noise: float = 0.0


class ImageLogResult(BaseModel):
    image_url: str
    scenario_type: str = "NORMAL"
    similarity_score: float = 0.0
    phash_score: float = 0.0
    foreground_similarity: float = 0.0
    background_similarity: float = 0.0
    quality_scores: QualityScores
    embedding_vector: List[float] = []


class ListingSummary(BaseModel):
    listing_id: str
    final_decision: str
    confidence_score: float
    reason: str


class DetectionResponse(BaseModel):
    status: str = "success"
    listing_summary: ListingSummary
    image_logs: List[ImageLogResult]


# ==========================================
# 3. API Endpoint (ต่อ Pipeline จริง)
# ==========================================
@app.post("/api/v1/detect-duplicate", response_model=DetectionResponse)
async def detect_duplicate_endpoint(payload: DetectionRequest):
    try:
        image_urls = payload.uploaded_images
        product_info = payload.uploaded_product_info

        if not image_urls:
            raise HTTPException(
                status_code=400, detail="No images provided in uploaded_images"
            )

        target_image_url = image_urls[0]

        # Step 1: โหลดภาพและคำนวณคะแนนคุณภาพ (Tech 2)
        img_rgb = load_image_from_url_or_path(target_image_url)
        quality_dict = calculate_quality_scores(img_rgb)

        # Step 2: สกัด pHash, YOLOv8 Mask, และ CLIP Embeddings (Tech 1)
        features = extract_all_features(img_rgb)
        new_phash = features["phash"]
        new_fg_vec = features["fg_vector"]
        new_bg_vec = features["bg_vector"]

        # Step 3: เปรียบเทียบกับภาพในฐานข้อมูล และตัดสินผล (Scoring Engine)
        decision, fg_sim, bg_sim, phash_sim_score, reason = evaluate_baseline_decision(
            new_fg_vec=new_fg_vec,
            new_bg_vec=new_bg_vec,
            new_phash=new_phash,
            db_records=DB_REFERENCE_IMAGES,
        )

        # Step 4: บันทึกข้อมูลภาพนี้ลง In-Memory DB เพื่อใช้เปรียบเทียบในรอบถัดไป
        DB_REFERENCE_IMAGES.append({
            "image_url": target_image_url,
            "phash": new_phash,
            "fg_vector": new_fg_vec,
            "bg_vector": new_bg_vec,
        })

        # Step 5: คืนค่า Response ตาม Data Contract
        return DetectionResponse(
            status="success",
            listing_summary=ListingSummary(
                listing_id=product_info.listing_id,
                final_decision=decision,
                confidence_score=fg_sim if decision == "DUPLICATE" else round(1.0 - fg_sim, 4),
                reason=reason,
            ),
            image_logs=[
                ImageLogResult(
                    image_url=target_image_url,
                    scenario_type="EXACT_DUPLICATE" if decision == "DUPLICATE" else "NORMAL",
                    similarity_score=fg_sim,
                    phash_score=phash_sim_score,
                    foreground_similarity=fg_sim,
                    background_similarity=bg_sim,
                    quality_scores=QualityScores(**quality_dict),
                    embedding_vector=new_fg_vec[:10],  # ส่งตัวอย่าง 10 มิติแรกเพื่อความกระชับ
                )
            ],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Pipeline processing failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)