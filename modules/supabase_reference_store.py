"""Supabase-backed reference and analysis store."""

import os
from typing import Dict, List, Optional
from uuid import uuid4

from supabase import Client, create_client

from modules.reference_store import ReferenceImageStore


class SupabaseReferenceStore(ReferenceImageStore):
    """Persist reference features and analysis results in Supabase."""

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        self._client: Client = create_client(
            url or os.environ["SUPABASE_URL"],
            key or os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )

    def find_candidates(
        self,
        new_fg_vector: List[float],
        new_bg_vector: List[float],
        new_phash: str,
        top_k: int = 20,
        scope: Optional[Dict] = None,
    ) -> List[Dict]:
        scope = scope or {}
        result = self._client.rpc(
            "match_product_images",
            {
                "query_vector": new_fg_vector,
                "match_count": top_k,
                "requested_seller_id": scope.get("seller_id"),
                "requested_category": scope.get("category"),
            },
        ).execute()
        return result.data or []

    def add(self, record: Dict) -> None:
        payload = {
            key: record[key]
            for key in (
                "use_case_id",
                "product_id",
                "listing_id",
                "image_id",
                "image_url",
                "seller_id",
                "category",
                "phash",
                "fg_vector",
                "bg_vector",
                "blur_score",
                "brightness_score",
                "contrast_score",
                "noise_score",
            )
            if key in record and record[key] is not None
        }
        self._client.table("product_reference_images").insert(payload).execute()

    def upload_image(
        self,
        contents: bytes,
        filename: Optional[str],
        content_type: Optional[str],
    ) -> str:
        bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "product-images")
        suffix = os.path.splitext(filename or "image.bin")[1].lower()
        storage_path = f"uploads/{uuid4()}{suffix}"
        self._client.storage.from_(bucket).upload(
            storage_path,
            contents,
            {"content-type": content_type or "application/octet-stream", "upsert": "false"},
        )
        return storage_path

    def record_analysis(self, result: Dict) -> None:
        payload = {
            key: result[key]
            for key in (
                "use_case_id",
                "uploaded_product_id",
                "uploaded_listing_id",
                "uploaded_image_id",
                "matched_reference_id",
                "decision",
                "is_repetition",
                "repetition_rate",
                "foreground_similarity",
                "background_similarity",
                "phash_similarity",
                "reason",
                "blur_score",
                "brightness_score",
                "contrast_score",
                "noise_score",
                "model_version",
            )
            if key in result and result[key] is not None
        }
        self._client.table("duplicate_analysis_results").insert(payload).execute()
