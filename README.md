# TTT AI Duplicate Detection Engine

FastAPI service for detecting duplicate product images using YOLOv8-Seg, OpenCLIP embeddings, and pHash.

The upload endpoint accepts JPEG/JPG, PNG, WebP, GIF, BMP, TIFF/TIF, AVIF, and
HEIC/HEIF images. HEIC/HEIF/AVIF decoding is provided by `pillow-heif`; the
service returns HTTP 400 for unsupported or corrupt image data.

## Architecture

```text
Uploaded image
    -> FastAPI multipart upload
    -> YOLOv8-Seg foreground/background segmentation
    -> OpenCLIP embeddings + pHash + quality scores
  -> reference-store candidate search (memory by default, Supabase pgvector when configured)
    -> Scoring decision
  -> optional Supabase Storage and analysis history
```

Supabase is used as the database and file storage only. The AI models run in the FastAPI process.

## Decisions

The scoring layer supports:

- `UNIQUE`
- `REVIEW`
- `DUPLICATE`
- `SPAM`
- `INVALID_DATA`

The live image API currently returns `UNIQUE`, `REVIEW`, `DUPLICATE`, or
`INVALID_DATA`. `SPAM` is implemented for the precomputed-dataset evaluator
and is allowed by the Supabase schema, but the live image scoring path does not
currently produce `SPAM`.

The current implementation is a conservative prototype. It is not yet calibrated against a real labeled image dataset.

The model scope is intentionally limited to phones/tablets, apparel/fashion, cameras, basic tools,
beauty/health products, and computer/IT accessories. Unsupported or uncertain detections return `REVIEW`.

## Current Prototype Status

The main inference and decision flow is implemented, but this repository does
not have a defensible percentage-complete or real-image accuracy measurement.

| Capability | Status | Notes |
|---|---|---|
| Image quality gate | Ready | Rejects clearly blurry, dark, low-contrast, or low-resolution images as `REVIEW` |
| YOLOv8n-Seg foreground/background | Implemented with fallback | Selects supported product-like masks and cleans masks; when segmentation is unavailable, the current extractor falls back to the full image instead of forcing `REVIEW` |
| OpenCLIP embedding | Ready for MVP | Generates normalized foreground/background vectors |
| pHash matching | Ready for MVP | Supports exact/near visual similarity checks |
| Decision scoring | Implemented with context gate | Live API returns `DUPLICATE`, `REVIEW`, `UNIQUE`, `INVALID_DATA`, or `SPAM`; `SPAM` requires `product_id`, `seller_id`, `listing_id`, `category`, and a duplicate reference from the same seller |
| Screenshot/watermark/AI-artifact gates | Basic | Conservative heuristic detectors; require real-image calibration |
| Real-image model validation | Pending | No labeled image results yet |
| Production identity/auth/persistence | Partial | Supabase reference/storage/history integration exists, but authentication, caller-provided product identity, and production validation are pending |

### Accuracy Currently Available

The repository dataset `dataset-ai-ตรวจสอบรูปภาพซ้ำ.xlsx` was checked with the evaluator's Phase 1 rules. It measures **decision logic from precomputed numeric signals**, not YOLO/OpenCLIP inference from real images:

| Split | Records | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Training | 20,000 | 100% | 100% |
| Validation | 2,000 | 100% | 100% |
| Testing | 2,000 | 100% | 100% |

These numbers confirm that the Phase 1 threshold rules classify the supplied numeric dataset correctly. They must not be reported as real-image model accuracy. Real-image accuracy is **not available yet** and is waiting for Stage B validation with Ground Truth images. No generated evaluation report is committed under `evaluation/`; generate one with the command below when needed.

### Waiting for Calibration

The following items require real labeled images before values can be considered final:

- YOLO mask confidence, mask size limits, supported-category behavior, and the current segmentation fallback behavior
- Foreground/background segmentation quality
- OpenCLIP foreground similarity threshold
- pHash distance threshold
- Quality-gate thresholds
- Screenshot, watermark, and AI-artifact detector thresholds
- Rotation, mirror, and crop behavior

Calibration workflow: run localhost/Docker performance first, then use the Calibration set, validate on the Validation set, lock configuration, and run the Final Testing set once. See [PERFORMANCE_AND_CALIBRATION_GUIDE.md](PERFORMANCE_AND_CALIBRATION_GUIDE.md).

### Possible Next Concepts

- Add `detected_category` as an optional output without making category classification a hard requirement
- Expose `segmentation_status` and `segmentation_reason` in the API response for QA observability
- Add a dedicated category classifier or zero-shot classifier if product category filtering becomes a business requirement
- Add rotation/flip-invariant matching if real-image results show a material failure rate
- Replace heuristic detectors with specialized models only when false-positive data justifies the cost
- Add a performance worker/queue when concurrency testing shows the synchronous inference path is saturated

## Requirements

- Windows
- Python 3.14 is the verified local runtime; the Dockerfile uses Python 3.11
- Supabase project with `vector` and `pgcrypto` extensions
- YOLO model file at `modules/yolov8n-seg.pt`
- Tesseract OCR executable on `PATH` (optional; required for OCR-based screenshot/watermark signals)

## Installation

Use the project virtual environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The archive intentionally excludes the local virtual environment, `.env`, Python cache files, and generated evaluation reports. The YOLO model file is included because the API loads it at startup.

## Supabase Setup

1. Create a Supabase project.
2. Open **SQL Editor**.
3. Run [db/schema.sql](db/schema.sql).
4. The schema creates:
   - `product_reference_images`
   - `duplicate_analysis_results`
   - `match_product_images()` RPC
5. Create a private Storage bucket named `product-images`.

The reference table stores vectors and image metadata. The analysis table stores each analysis result separately.

## Environment

Copy `.env.example` to `.env` and set the real server key:

```env
REFERENCE_STORE=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_STORAGE_BUCKET=product-images
YOLO_SEG_CONFIDENCE=0.25
YOLO_SEG_IOU=0.7
YOLO_SEG_MASK_CONFIDENCE=0.35
YOLO_SEG_MAX_OBJECTS=3
```

`SUPABASE_SERVICE_ROLE_KEY` is server-only. Never expose it in frontend code, logs, screenshots, or Git. If a key has been shared publicly, rotate it in Supabase before use.

`YOLO_SEG_MASK_CONFIDENCE` controls the minimum confidence for a selected segmentation mask.
`YOLO_SEG_MAX_OBJECTS` limits the number of product masks combined into the foreground.

For local development without Supabase:

```env
REFERENCE_STORE=memory
```

## Run the API

```powershell
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or run [start_server_ngrok.bat](start_server_ngrok.bat). The batch file currently starts FastAPI only; it does not start an ngrok process.

Open Swagger UI:

```text
http://localhost:8000/docs
```

Choose **Try it out**, select an image in the `image` field, and execute the request.

Endpoint:

```text
POST /gateway/detectDuplicateProduct
Content-Type: multipart/form-data
```

Allowed formats are JPEG, PNG, and WebP. Maximum upload size is 10 MB.

## Example Response

```json
{
  "status": "success",
  "is_repetition": true,
  "repetition_rate": 0.98,
  "matched_product": {
    "product_id": "product-001",
    "image_reference": "uploads/example.jpg"
  },
  "analysis": {
    "decision": "DUPLICATE",
    "reason": "Duplicate image detected",
    "similarity_breakdown": {
      "repetition_similarity": 0.98,
      "foreground_similarity": 0.98,
      "background_similarity": 0.95,
      "phash_similarity": 1.0
    },
    "quality_scores": {
      "blur": 0.8,
      "brightness": 0.5,
      "contrast": 0.7,
      "noise": 0.1
    }
  }
}
```

With `REFERENCE_STORE=supabase`, `image_reference` is a Storage path returned as response metadata. With the default in-memory store, it is the uploaded filename. The uploaded file itself is sent through the multipart `image` field.

## Evaluation

The evaluation harness uses precomputed numeric signals from the Excel/CSV dataset. It does not run YOLO or OpenCLIP inference.

For localhost FastAPI performance testing, use [QA_VALIDATION_TESTING_DATASET_REQUIREMENTS.md](QA_VALIDATION_TESTING_DATASET_REQUIREMENTS.md). It defines the workload images, concurrency scenarios, latency/throughput metrics, resource measurements, and QA result format. This test does not measure model accuracy.

After performance passes, use [PERFORMANCE_AND_CALIBRATION_GUIDE.md](PERFORMANCE_AND_CALIBRATION_GUIDE.md) and [VALIDATION_CALIBRATION_DATASET_REQUIREMENTS.md](VALIDATION_CALIBRATION_DATASET_REQUIREMENTS.md) for external access, Ground Truth validation, reporting, and model calibration.

```powershell
.venv\Scripts\python.exe evaluation\evaluate_dataset.py path\to\dataset.xlsx --output evaluation\report.json
```

The report includes legacy and Phase 1 metrics for Training, Validation, and Testing splits.

## Verification

Compile the project:

```powershell
.venv\Scripts\python.exe -m compileall -q main.py modules evaluation
```

Check Supabase tables:

```sql
select * from product_reference_images order by created_at desc limit 5;
select * from duplicate_analysis_results order by created_at desc limit 5;
```

## Current Limitations

- `InMemoryReferenceStore` is the default, returns all in-memory records rather than applying a real `top_k` nearest-neighbor search, and loses data when the process restarts.
- YOLOv8n-Seg is COCO-pretrained and may not detect all e-commerce products.
- A failed or unsupported segmentation currently falls back to embedding the full image; the API does not yet expose `segmentation_status` or force that case to `REVIEW`.
- Screenshot, watermark, and AI-artifact detectors are connected to the live pipeline as conservative heuristic gates; a flag returns `REVIEW`.
- Stock image detection is intentionally out of scope because there is no stock reference database.
- Thresholds are initial prototype values and require validation with real labeled images.
- The current API still uses the uploaded filename as the temporary product identifier.
- Authentication and integration with `public.ocr` are not implemented. The `api-caller` and `api-key` headers are accepted but are not validated.

## ฉบับภาษาไทย

### ภาพรวม

โปรเจกต์นี้เป็นบริการ FastAPI สำหรับตรวจจับภาพสินค้าที่ซ้ำกัน โดยใช้ YOLOv8-Seg แยก foreground/background, OpenCLIP สร้าง image embedding และ pHash เปรียบเทียบความใกล้เคียงของภาพ

Supabase ทำหน้าที่เป็น database และ file storage เท่านั้น ส่วนโมเดล AI ทำงานอยู่ใน FastAPI

### ลำดับการทำงาน

```text
อัปโหลดไฟล์ภาพ
  -> FastAPI รับ multipart/form-data
  -> YOLOv8-Seg แยก foreground/background
  -> OpenCLIP + pHash + quality scores
  -> ค้นหา candidate จาก Supabase pgvector
  -> scoring engine ตัดสินผล
  -> เก็บไฟล์และผลวิเคราะห์ใน Supabase
```

ผลลัพธ์ที่รองรับมี 5 ประเภท:

- `UNIQUE` ภาพไม่ซ้ำ
- `REVIEW` มีความคล้ายสูง ควรตรวจสอบเพิ่มเติม
- `DUPLICATE` พบภาพซ้ำ
- `SPAM` เข้าข่ายการอัปโหลดซ้ำในลักษณะ spam
- `INVALID_DATA` ข้อมูลภาพหรือ feature ไม่ถูกต้อง

### สิ่งที่ต้องเตรียม

- Windows
- Python และ virtual environment ในโฟลเดอร์ `.venv`
- Supabase project
- YOLO model ที่ `modules/yolov8n-seg.pt`

### การติดตั้ง

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### การตั้งค่า Supabase

1. เปิด Supabase Dashboard และเข้า **SQL Editor**
2. รันไฟล์ [db/schema.sql](db/schema.sql)
3. ตรวจว่ามีตาราง `product_reference_images` และ `duplicate_analysis_results`
4. ตรวจว่ามี RPC ชื่อ `match_product_images()`
5. สร้าง Storage bucket ชื่อ `product-images`
6. แนะนำให้ตั้ง bucket เป็น Private

ตาราง `product_reference_images` ใช้เก็บข้อมูลอ้างอิงและ vector สำหรับค้นหา ส่วน `duplicate_analysis_results` ใช้เก็บผลวิเคราะห์แต่ละครั้ง

### ไฟล์ Environment

คัดลอก `.env.example` เป็น `.env` แล้วใส่ค่าจริง:

```env
REFERENCE_STORE=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_STORAGE_BUCKET=product-images
YOLO_SEG_CONFIDENCE=0.25
YOLO_SEG_IOU=0.7
YOLO_SEG_MASK_CONFIDENCE=0.35
YOLO_SEG_MAX_OBJECTS=3
```

`SUPABASE_SERVICE_ROLE_KEY` ใช้เฉพาะฝั่ง backend ห้ามใส่ใน frontend, log, screenshot หรือ commit เข้า Git หาก key ถูกเปิดเผยแล้ว ให้ rotate key ใน Supabase ก่อนใช้งาน

`YOLO_SEG_MASK_CONFIDENCE` กำหนด confidence ขั้นต่ำของ mask ที่จะนำมาใช้ และ `YOLO_SEG_MAX_OBJECTS` จำกัดจำนวน mask สินค้าที่นำมารวมเป็น foreground

ถ้าต้องการทดสอบโดยไม่เชื่อม Supabase:

```env
REFERENCE_STORE=memory
```

### การรันระบบ

```powershell
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

หรือรัน [start_server_ngrok.bat](start_server_ngrok.bat)

เปิด Swagger UI ที่:

```text
http://localhost:8000/docs
```

กด **Try it out** เลือกไฟล์ในช่อง `image` แล้วกด Execute

Endpoint:

```text
POST /gateway/detectDuplicateProduct
Content-Type: multipart/form-data
```

รองรับ JPEG, PNG และ WebP ขนาดไม่เกิน 10 MB

ไฟล์ที่อัปโหลดจะถูกเก็บใน Supabase Storage ที่ path รูปแบบ:

```text
uploads/<uuid>.<extension>
```

### การประเมินผล

evaluation harness ใช้ค่า feature ที่คำนวณไว้แล้วจาก dataset จึงไม่ได้รัน YOLO หรือ OpenCLIP จริง:

สำหรับการทดสอบ performance ของ FastAPI บน localhost ให้ใช้ [QA_VALIDATION_TESTING_DATASET_REQUIREMENTS.md](QA_VALIDATION_TESTING_DATASET_REQUIREMENTS.md) ซึ่งกำหนด workload ภาพ, concurrency, latency, throughput, resource usage และ format ผลทดสอบ โดยรอบนี้ไม่วัด accuracy ของโมเดล

เมื่อ performance ผ่านแล้ว ให้ใช้ [PERFORMANCE_AND_CALIBRATION_GUIDE.md](PERFORMANCE_AND_CALIBRATION_GUIDE.md) และ [VALIDATION_CALIBRATION_DATASET_REQUIREMENTS.md](VALIDATION_CALIBRATION_DATASET_REQUIREMENTS.md) สำหรับการเปิดให้ QA ภายนอกเข้าทดสอบ, การส่ง report และการ calibrate โมเดล

```powershell
.venv\Scripts\python.exe evaluation\evaluate_dataset.py path\to\dataset.xlsx --output evaluation\report.json
```

รายงานจะแสดง Accuracy, Precision, Recall, F1 และผลแยกตาม scenario ของ Training, Validation และ Testing

### การตรวจสอบระบบ

ตรวจ syntax:

```powershell
.venv\Scripts\python.exe -m compileall -q main.py modules evaluation
```

ตรวจข้อมูลใน Supabase:

```sql
select * from product_reference_images order by created_at desc limit 5;
select * from duplicate_analysis_results order by created_at desc limit 5;
```

### ขอบเขตโมเดลต้นแบบ

ระบบโฟกัสเฉพาะสินค้า:

- มือถือและแท็บเล็ต
- เสื้อผ้าและแฟชั่น
- กล้อง
- อุปกรณ์เครื่องมือพื้นฐาน
- ความงามและสุขภาพ
- คอมพิวเตอร์และ IT เช่น laptop, monitor, keyboard, mouse, router, speaker และ headphones

YOLOv8n-Seg จะคัด mask ตาม class, confidence และขนาดวัตถุ แล้วทำความสะอาด mask ก่อนแยก foreground/background
ด้วย morphological filtering หากไม่พบ mask ที่น่าเชื่อถือ ระบบจะคืน `REVIEW`

Screenshot, watermark และ AI-artifact detector ทำงานใน live pipeline แบบ conservative เช่นกัน โดยไม่ตัดสิน `DUPLICATE`
จาก detector เพียงอย่างเดียว

### ข้อจำกัดปัจจุบัน

- โหมด `InMemoryReferenceStore` ข้อมูลจะหายเมื่อ restart server
- YOLOv8n-Seg เป็นโมเดลที่ train จาก COCO จึงอาจตรวจจับสินค้าบางประเภทไม่ได้
- detector ปัจจุบันเป็น heuristic ยังไม่ใช่โมเดลเฉพาะทาง และต้อง validate false positive กับภาพจริง
- ยังไม่ทำ Stock image detection เพราะไม่มีฐานข้อมูลภาพ stock อ้างอิง
- threshold ปัจจุบันเป็นค่า prototype: duplicate `>= 0.95`, review `>= 0.70`
- ยังไม่มีชุดภาพจริงพร้อม Ground Truth สำหรับวัด pipeline ตั้งแต่ YOLO ถึง decision
- API ยังใช้ชื่อไฟล์เป็น product identifier ชั่วคราว
- ยังไม่มี authentication และการเชื่อมต่อกับ `public.ocr`

### สถานะ Prototype และเปอร์เซ็นต์ความคืบหน้า

โค้ดมี flow หลักของ model prototype แล้ว แต่ยังไม่มีเปอร์เซ็นต์ความคืบหน้าหรือความแม่นยำจากภาพจริงที่ยืนยันได้

| ความสามารถ | สถานะ | รายละเอียด |
|---|---|---|
| Quality gate | พร้อมใช้ | คัดภาพเบลอ มืด contrast ต่ำ และความละเอียดต่ำเป็น `REVIEW` |
| YOLOv8n-Seg | มี implementation พร้อม fallback | เลือก mask สินค้าที่เหมาะสมและทำความสะอาด mask; หาก segmentation ไม่ผ่าน จะใช้ full image ต่อ ไม่ได้คืน `REVIEW` โดยอัตโนมัติ |
| OpenCLIP embedding | พร้อมใช้ระดับ MVP | สร้าง vector ของ foreground/background |
| pHash matching | พร้อมใช้ระดับ MVP | ตรวจความเหมือนของภาพแบบ exact/near |
| Decision scoring | มี implementation | live API ใช้ `DUPLICATE`, `REVIEW`, `UNIQUE`, `INVALID_DATA`; `SPAM` ใช้ใน evaluator เท่านั้น |
| Screenshot/watermark/AI artifact | พื้นฐาน | เป็น heuristic และยังต้อง calibrate ด้วยภาพจริง |
| Validation ด้วยภาพจริง | รอทำ | ยังไม่มีผลจาก Ground Truth dataset |
| Production integration | ทำบางส่วน | มี Supabase reference/storage/history integration แต่ identity, auth และ production validation ยังไม่เสร็จ |

### ความแม่นยำที่มีตอนนี้

จากการรันกับไฟล์ `dataset-ai-ตรวจสอบรูปภาพซ้ำ.xlsx` ใน repository, evaluation วัดเฉพาะ **decision logic จาก numeric signals ที่คำนวณไว้แล้ว** ไม่ได้วัด YOLO หรือ OpenCLIP จากภาพจริง:

| ชุดข้อมูล | จำนวน records | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Training | 20,000 | 100% | 100% |
| Validation | 2,000 | 100% | 100% |
| Testing | 2,000 | 100% | 100% |

ตัวเลขนี้ยืนยันว่า threshold rules ของ Phase 1 จัดกลุ่ม numeric dataset ที่มีอยู่ได้ถูกต้อง แต่ห้ามตีความเป็นความแม่นยำของโมเดลกับภาพจริง เพราะ real-image accuracy ยังไม่มีผลทดสอบและกำลังรอ Stage B validation พร้อม Ground Truth

### ส่วนที่รอ Calibration

- YOLO mask confidence, ขนาด mask และ supported-category behavior
- คุณภาพการแยก foreground/background
- foreground similarity threshold ของ OpenCLIP
- pHash distance threshold
- quality-gate thresholds
- screenshot, watermark และ AI-artifact detector thresholds
- พฤติกรรมภาพหมุน กลับด้าน และ crop

ลำดับคือรัน performance บน localhost/Docker ก่อน จากนั้น calibrate ด้วย Calibration set, ตรวจซ้ำด้วย Validation set, lock config แล้วรัน Final Testing เพียงครั้งเดียว รายละเอียดอยู่ใน [PERFORMANCE_AND_CALIBRATION_GUIDE.md](PERFORMANCE_AND_CALIBRATION_GUIDE.md)

### แนวคิดที่อาจทำต่อ

- เพิ่ม `detected_category` เป็น output แบบ optional โดยไม่บังคับให้ category classification เป็นเงื่อนไขหลัก
- เปิด `segmentation_status` และ `segmentation_reason` ใน API response เพื่อให้ QA ตรวจสอบได้
- เพิ่ม category classifier หรือ zero-shot classifier หาก business ต้องการกรอง category จริง
- เพิ่ม rotation/flip-invariant matching หากผลภาพจริงพบปัญหามากพอ
- เปลี่ยน heuristic detector เป็น specialized model เมื่อมีข้อมูล false positive รองรับ
- เพิ่ม worker/queue หาก performance test พบว่า inference แบบ synchronous รองรับ concurrency ไม่พอ
