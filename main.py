from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Duplicate Image Detection Engine",
    description="Phase 1: Baseline Single-Image Duplicate Detection API",
    version="1.0.0",
)


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
# 3. API Endpoint
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

        # ข้อมูล Mock Data เพื่อทดสอบ API ในวันแรก
        return DetectionResponse(
            status="success",
            listing_summary=ListingSummary(
                listing_id=product_info.listing_id,
                final_decision="UNIQUE",
                confidence_score=0.95,
                reason="Phase 1 Baseline passed - No duplicate found",
            ),
            image_logs=[
                ImageLogResult(
                    image_url=target_image_url,
                    scenario_type="NORMAL",
                    similarity_score=0.15,
                    phash_score=0.10,
                    foreground_similarity=0.12,
                    background_similarity=0.18,
                    quality_scores=QualityScores(
                        blur=0.05, brightness=0.48, contrast=0.58, noise=0.12
                    ),
                    embedding_vector=[0.012, -0.045, 0.891],
                )
            ],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Pipeline processing failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)