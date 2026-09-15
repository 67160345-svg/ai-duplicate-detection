import logging
import os
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import uvicorn

# นำเข้าโมดูล AI ภายในระบบ
from modules.feature_extractor import extract_all_features
from modules.preprocessor import calculate_quality_scores, check_image_quality, load_image_from_bytes
from modules.scoring_engine import (
    DEFAULT_SIMILARITY_THRESHOLDS,
    evaluate_spam_decision,
    evaluate_baseline_decision,
)
from modules.detectors import run_detectors
from modules.reference_store import ReferenceImageStore, InMemoryReferenceStore
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

logger = logging.getLogger("ttt_ai_engine")

app = FastAPI(
    title="TTT AI Duplicate Detection Engine",
    description="Prototype: SE Gateway Integration (In-Memory Reference)",
    version="1.0.0-beta",
)


def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["openapi"] = "3.0.3"
    upload_schema = schema["components"]["schemas"].get(
        "Body_detect_duplicate_product_gateway_gateway_detectDuplicateProduct_post"
    )
    if upload_schema:
        image_schema = upload_schema.get("properties", {}).get("image")
        if image_schema:
            image_schema.pop("contentMediaType", None)
            image_schema["format"] = "binary"
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi

def _create_reference_store() -> ReferenceImageStore:
    if os.getenv("REFERENCE_STORE", "memory").lower() == "supabase":
        try:
            from modules.supabase_reference_store import SupabaseReferenceStore
        except ImportError as exc:
            raise RuntimeError(
                "Supabase mode requires the 'supabase' package"
            ) from exc
        return SupabaseReferenceStore()
    return InMemoryReferenceStore()


reference_store: ReferenceImageStore = _create_reference_store()

# จำกัดขนาดไฟล์อัปโหลดกันคนส่งไฟล์ใหญ่มาถล่ม (ปรับตามความเหมาะสมของ use case จริง)
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {
    "image/avif",
    "image/bmp",
    "image/gif",
    "image/heic",
    "image/heif",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "image/webp",
}
ALLOWED_IMAGE_EXTENSIONS = {
    ".avif", ".bmp", ".gif", ".heic", ".heif", ".jpeg", ".jpg",
    ".png", ".tif", ".tiff", ".webp",
}


def _clean_optional_form_value(value: Optional[str]) -> Optional[str]:
    """Treat Swagger placeholder values as missing optional request context."""
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned.lower() in {"", "string", "null", "none"}:
        return None
    return cleaned


# ==========================================
# 1. Pydantic Response Contract (SE Specifications)
# ==========================================
class MatchedProductInfo(BaseModel):
    product_id: Optional[str] = None
    image_reference: Optional[str] = None


class SimilarityBreakdown(BaseModel):
    # หมายเหตุ: repetition_similarity คือค่าเดียวกับ repetition_rate ที่ระดับบนสุด
    # (fg_sim เมื่อ decision=DUPLICATE, บังคับเป็น 0.0 เมื่อ decision=UNIQUE)
    # ไม่ใช่คะแนนรวมของหลาย signal — ถ้าต้องการดูค่า fg_sim ดิบๆ เสมอไม่ว่า
    # decision จะเป็นอะไร ให้ดูที่ foreground_similarity แทน
    repetition_similarity: float
    foreground_similarity: float
    background_similarity: float
    phash_similarity: float


class AnalysisDetail(BaseModel):
    decision: str
    reason: str
    similarity_breakdown: SimilarityBreakdown
    quality_scores: Dict[str, float]


class GatewayDetectionResponse(BaseModel):
    status: str = "success"
    is_repetition: bool
    repetition_rate: float
    matched_product: Optional[MatchedProductInfo] = None
    analysis: AnalysisDetail


# ==========================================
# 2. Gateway Endpoint
# ==========================================
@app.post(
    "/gateway/detectDuplicateProduct", response_model=GatewayDetectionResponse
)
async def detect_duplicate_product_gateway(
    image: UploadFile = File(...),
    product_id: Optional[str] = Form(None),
    seller_id: Optional[str] = Form(None),
    listing_id: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    api_caller: str = Header(None, alias="api-caller"),
    api_key: str = Header(None, alias="api-key"),
):
    product_id = _clean_optional_form_value(product_id)
    seller_id = _clean_optional_form_value(seller_id)
    listing_id = _clean_optional_form_value(listing_id)
    category = _clean_optional_form_value(category)

    try:
        # Step 0: ตรวจสอบไฟล์เบื้องต้นก่อนเข้า pipeline (กัน DoS / ไฟล์ผิดชนิด)
        image_extension = Path(image.filename or "").suffix.lower()
        if (
            image.content_type not in ALLOWED_CONTENT_TYPES
            and image_extension not in ALLOWED_IMAGE_EXTENSIONS
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "ไฟล์ต้องเป็นรูปภาพประเภท "
                    f"{', '.join(sorted(ALLOWED_CONTENT_TYPES))} เท่านั้น"
                ),
            )

        # Step 1: อ่าน Raw Bytes จาก UploadFile และแปลงเป็น RGB Matrix
        contents = await image.read()
        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "ไฟล์มีขนาดเกิน "
                    f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB"
                ),
            )

        # load_image_from_bytes / extract_all_features เป็นงาน CPU-bound
        # แบบ synchronous (YOLO/CLIP inference) — ต้องรันใน threadpool ไม่งั้น
        # จะบล็อก event loop ทำให้ request อื่นค้างหมดระหว่างประมวลผลภาพนี้
        try:
            img_rgb = await run_in_threadpool(load_image_from_bytes, contents)
        except ValueError as exc:
            logger.exception(
                "Image decode failed: filename=%r content_type=%r bytes=%d",
                image.filename,
                image.content_type,
                len(contents),
            )
            raise HTTPException(
                status_code=400,
                detail="ไฟล์ไม่ใช่ภาพที่รองรับหรือภาพเสียหาย",
            ) from exc

        quality_scores = await run_in_threadpool(calculate_quality_scores, img_rgb)
        quality_ok, quality_issues = await run_in_threadpool(check_image_quality, img_rgb)
        if not quality_ok:
            reason = "ภาพไม่พร้อมสำหรับ detection: " + "; ".join(quality_issues)
            return GatewayDetectionResponse(
                status="success",
                is_repetition=False,
                repetition_rate=0.0,
                matched_product=None,
                analysis=AnalysisDetail(
                    decision="REVIEW",
                    reason=reason,
                    similarity_breakdown=SimilarityBreakdown(
                        repetition_similarity=0.0,
                        foreground_similarity=0.0,
                        background_similarity=0.0,
                        phash_similarity=0.0,
                    ),
                    quality_scores=quality_scores,
                ),
            )

        # Phase 3 detectors: conservative, soft review flags.
        detector_results = await run_in_threadpool(run_detectors, img_rgb)
        detector_reasons = [
            f"{name}: {info['reason']}"
            for name, info in detector_results.items()
            if info.get("flagged")
        ]
        if detector_reasons:
            reason = "; ".join(detector_reasons)
            return GatewayDetectionResponse(
                status="success",
                is_repetition=False,
                repetition_rate=0.0,
                matched_product=None,
                analysis=AnalysisDetail(
                    decision="REVIEW",
                    reason=f"ภาพมีสัญญาณไม่เหมาะสำหรับการเทียบซ้ำ: {reason}",
                    similarity_breakdown=SimilarityBreakdown(
                        repetition_similarity=0.0,
                        foreground_similarity=0.0,
                        background_similarity=0.0,
                        phash_similarity=0.0,
                    ),
                    quality_scores=quality_scores,
                ),
            )

        # Step 2: สกัด Features (YOLOv8-Seg + OpenCLIP + pHash)
        features = await run_in_threadpool(extract_all_features, img_rgb)
        new_phash = features["phash"]
        new_fg_vec = features["fg_vector"]
        new_bg_vec = features["bg_vector"]

        # Step 3: ดึง candidate ที่ใกล้เคียงที่สุดจาก store มาเทียบแบบละเอียด
        # (ตอนนี้ store เป็น in-memory จึงคืนทั้งหมด — พอสลับเป็น pgvector
        # ขั้นนี้จะคืนแค่ top_k ที่ index กรองมาให้แล้ว ไม่ต้องแก้โค้ดตรงนี้)
        candidates = await run_in_threadpool(
            reference_store.find_candidates,
            new_fg_vector=new_fg_vec,
            new_bg_vector=new_bg_vec,
            new_phash=new_phash,
        )
        decision, fg_sim, bg_sim, phash_sim, reason, matched = (
            evaluate_baseline_decision(
                new_fg_vec=new_fg_vec,
                new_bg_vec=new_bg_vec,
                new_phash=new_phash,
                db_records=candidates,
                fg_threshold=DEFAULT_SIMILARITY_THRESHOLDS["near_duplicate"],
                phash_distance_threshold=5,
                review_threshold=DEFAULT_SIMILARITY_THRESHOLDS["review"],
            )
        )

        spam_context = {
            "product_id": product_id,
            "seller_id": seller_id,
            "listing_id": listing_id,
            "category": category,
        }
        decision, spam_reason = evaluate_spam_decision(
            context=spam_context,
            base_decision=decision,
            matched_record=matched,
        )
        if spam_reason:
            reason = spam_reason

        is_rep = decision in {"DUPLICATE", "SPAM"}
        rep_rate = fg_sim if decision in {"DUPLICATE", "REVIEW"} else 0.0

        image_reference = image.filename
        if hasattr(reference_store, "upload_image"):
            image_reference = await run_in_threadpool(
                reference_store.upload_image,
                contents,
                image.filename,
                image.content_type,
            )

        # Step 4: บันทึกเฉพาะ feature ที่ผ่าน validation เพื่อไม่ให้ข้อมูลเสีย
        # กลายเป็น reference สำหรับ request ถัดไป
        if decision != "INVALID_DATA":
            current_record = {
                "product_id": product_id or image.filename,
                "image_url": image_reference,
                "phash": new_phash,
                "fg_vector": new_fg_vec,
                "bg_vector": new_bg_vec,
            }
            if seller_id:
                current_record["seller_id"] = seller_id
            if listing_id:
                current_record["listing_id"] = listing_id
            if category:
                current_record["category"] = category
            await run_in_threadpool(reference_store.add, current_record)

        if hasattr(reference_store, "record_analysis"):
            await run_in_threadpool(
                reference_store.record_analysis,
                {
                    "uploaded_product_id": product_id or image.filename,
                    "seller_id": seller_id,
                    "listing_id": listing_id,
                    "category": category,
                    "business_rule": "BR010" if decision == "SPAM" else None,
                    "seller_match": decision == "SPAM",
                    "spam_reason": spam_reason,
                    "matched_reference_id": matched.get("id") if matched else None,
                    "decision": decision,
                    "is_repetition": is_rep,
                    "repetition_rate": rep_rate,
                    "foreground_similarity": fg_sim,
                    "background_similarity": bg_sim,
                    "phash_similarity": phash_sim,
                    "reason": reason,
                    "blur_score": quality_scores["blur"],
                    "brightness_score": quality_scores["brightness"],
                    "contrast_score": quality_scores["contrast"],
                    "noise_score": quality_scores["noise"],
                    "model_version": "yolov8n-seg-openclip-vit-b-32-v1",
                },
            )

        # Step 5: คืนค่า Response ตาม Contract
        return GatewayDetectionResponse(
            status="success",
            is_repetition=is_rep,
            repetition_rate=rep_rate,
            matched_product=(
                MatchedProductInfo(
                    product_id=matched.get("product_id") if matched else None,
                    image_reference=matched.get("image_url")
                    if matched
                    else None,
                )
                if decision in {"DUPLICATE", "REVIEW", "SPAM"} and matched
                else None
            ),
            analysis=AnalysisDetail(
                decision=decision,
                reason=reason,
                similarity_breakdown=SimilarityBreakdown(
                    repetition_similarity=rep_rate,
                    foreground_similarity=fg_sim,
                    background_similarity=bg_sim,
                    phash_similarity=phash_sim,
                ),
                quality_scores=quality_scores,
            ),
        )

    except HTTPException:
        # 400/413 ที่ raise เองด้านบน ให้ผ่านไปตามเดิม ไม่ใช่ไป wrap เป็น 500
        raise
    except Exception:
        # Log stack trace เต็มๆ ไว้ฝั่ง server เพื่อ debug แต่ตอบ client แบบ
        # generic เท่านั้น กัน internal detail (path, library error) หลุดออกไป
        logger.exception("Pipeline processing failed")
        raise HTTPException(
            status_code=500,
            detail="Pipeline processing failed. Please try again or contact support.",
        )


if __name__ == "__main__":
    # reload=True เหมาะแค่ dev เท่านั้น ใน production ต้องปิด (default ปิดไว้แล้ว
    # เปิดได้ด้วย env var ENVIRONMENT=development)
    is_dev = os.getenv("ENVIRONMENT", "production").lower() == "development"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=is_dev)