-- schema.sql
--
-- Schema อ้างอิงสำหรับทีม API เพื่อ implement PgVectorReferenceStore
-- (ตาม interface ใน reference_store.py) — ยังไม่ได้รันจริง เป็นแค่ spec
--
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE product_reference_images (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Input/reference data for each use case.
    use_case_id     TEXT,
    product_id      TEXT NOT NULL,
    listing_id      TEXT,
    image_id        TEXT,
    image_url       TEXT,

    -- เผื่อ business ต้องการ scope การเทียบแบบ "เฉพาะร้านเดียวกัน"
    -- (ยังไม่ยืนยัน — ใส่ไว้เป็น nullable ก่อน ถ้าไม่ใช้ก็ปล่อย NULL ทั้งคอลัมน์)
    seller_id       TEXT,
    category        TEXT,

    phash           TEXT NOT NULL,
    fg_vector       VECTOR(512) NOT NULL,  -- ขนาดตาม OpenCLIP ViT-B-32
    bg_vector       VECTOR(512) NOT NULL,

    blur_score      REAL,
    brightness_score REAL,
    contrast_score  REAL,
    noise_score     REAL,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index สำหรับ approximate nearest neighbor บน cosine distance
-- lists ควรปรับตามขนาดข้อมูลจริง (ค่าประมาณ: sqrt(จำนวนแถวทั้งหมด))
CREATE INDEX product_reference_images_fg_vector_idx
    ON product_reference_images
    USING ivfflat (fg_vector vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX product_reference_images_bg_vector_idx
    ON product_reference_images
    USING ivfflat (bg_vector vector_cosine_ops)
    WITH (lists = 100);

-- ถ้าตัดสินใจใช้ scope ต่อร้าน ควรมี index รองรับการ filter ก่อน vector search
CREATE INDEX product_reference_images_seller_id_idx
    ON product_reference_images (seller_id);

CREATE INDEX product_reference_images_use_case_id_idx
    ON product_reference_images (use_case_id);

-- Result data is separated from reference data so one input can have multiple
-- analysis attempts without changing the vectors used for retrieval.
CREATE TABLE duplicate_analysis_results (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    use_case_id             TEXT,
    uploaded_product_id     TEXT,
    uploaded_listing_id     TEXT,
    uploaded_image_id       TEXT,
    matched_reference_id    UUID REFERENCES product_reference_images(id),

    decision                TEXT NOT NULL CHECK (
        decision IN ('UNIQUE', 'REVIEW', 'DUPLICATE', 'SPAM', 'INVALID_DATA')
    ),
    is_repetition           BOOLEAN NOT NULL DEFAULT false,
    repetition_rate         REAL NOT NULL DEFAULT 0.0,
    foreground_similarity   REAL NOT NULL DEFAULT 0.0,
    background_similarity   REAL NOT NULL DEFAULT 0.0,
    phash_similarity        REAL NOT NULL DEFAULT 0.0,
    reason                  TEXT,

    blur_score              REAL,
    brightness_score        REAL,
    contrast_score          REAL,
    noise_score             REAL,
    model_version           TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX duplicate_analysis_results_use_case_id_idx
    ON duplicate_analysis_results (use_case_id);

CREATE INDEX duplicate_analysis_results_decision_idx
    ON duplicate_analysis_results (decision);

-- Supabase RPC used by PgVectorReferenceStore.
CREATE OR REPLACE FUNCTION match_product_images(
    query_vector VECTOR(512),
    match_count INTEGER DEFAULT 20,
    requested_seller_id TEXT DEFAULT NULL,
    requested_category TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    product_id TEXT,
    listing_id TEXT,
    image_id TEXT,
    image_url TEXT,
    phash TEXT,
    fg_vector VECTOR(512),
    bg_vector VECTOR(512)
)
LANGUAGE SQL
STABLE
AS $$
    SELECT
        pri.id,
        pri.product_id,
        pri.listing_id,
        pri.image_id,
        pri.image_url,
        pri.phash,
        pri.fg_vector,
        pri.bg_vector
    FROM product_reference_images AS pri
    WHERE (requested_seller_id IS NULL OR pri.seller_id = requested_seller_id)
      AND (requested_category IS NULL OR pri.category = requested_category)
    ORDER BY pri.fg_vector <=> query_vector
    LIMIT match_count;
$$;

-- ตัวอย่าง query ที่ PgVectorReferenceStore.find_candidates() ควรทำ
-- (top_k=20, ไม่ filter scope):
--
-- SELECT product_id, image_url, phash, fg_vector, bg_vector
-- FROM product_reference_images
-- ORDER BY fg_vector <=> $1  -- $1 = new_fg_vector
-- LIMIT 20;
--
-- ถ้าต้อง filter ตาม seller_id (เมื่อ scope ถูกกำหนดแล้ว):
--
-- SELECT product_id, image_url, phash, fg_vector, bg_vector
-- FROM product_reference_images
-- WHERE seller_id = $2
-- ORDER BY fg_vector <=> $1
-- LIMIT 20;
