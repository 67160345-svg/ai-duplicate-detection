# TTT AI Duplicate Detection Engine

FastAPI service for detecting duplicate product images using YOLOv8-Seg, OpenCLIP embeddings, and pHash.

## Architecture

```text
Uploaded image
    -> FastAPI multipart upload
    -> YOLOv8-Seg foreground/background segmentation
    -> OpenCLIP embeddings + pHash + quality scores
    -> Supabase pgvector candidate search
    -> Scoring decision
    -> Supabase Storage and analysis history
```

Supabase is used as the database and file storage only. The AI models run in the FastAPI process.

## Decisions

The scoring layer supports:

- `UNIQUE`
- `REVIEW`
- `DUPLICATE`
- `SPAM`
- `INVALID_DATA`

The current implementation is a first YOLO version. It is not yet validated against a real image dataset.

## Requirements

- Windows
- Python 3.14 or compatible Python version
- Supabase project with `vector` and `pgcrypto` extensions
- YOLO model file at `modules/yolov8n-seg.pt`

## Installation

Use the project virtual environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

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
```

`SUPABASE_SERVICE_ROLE_KEY` is server-only. Never expose it in frontend code, logs, screenshots, or Git. If a key has been shared publicly, rotate it in Supabase before use.

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

`image_reference` is a Storage path returned as response metadata. The uploaded file itself is sent through the multipart `image` field.

## Evaluation

The evaluation harness uses precomputed numeric signals from the Excel/CSV dataset. It does not run YOLO or OpenCLIP inference.

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

- `InMemoryReferenceStore` loses data when the process restarts.
- YOLOv8n-Seg is COCO-pretrained and may not detect all e-commerce products.
- Watermark, screenshot, AI-image, and stock-image detectors are not implemented in the live image pipeline.
- Thresholds are baseline values and require validation with real labeled images.
- The current API still uses the uploaded filename as the temporary product identifier.
- Authentication and integration with `public.ocr` are not implemented.

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
```

`SUPABASE_SERVICE_ROLE_KEY` ใช้เฉพาะฝั่ง backend ห้ามใส่ใน frontend, log, screenshot หรือ commit เข้า Git หาก key ถูกเปิดเผยแล้ว ให้ rotate key ใน Supabase ก่อนใช้งาน

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

### ข้อจำกัดปัจจุบัน

- โหมด `InMemoryReferenceStore` ข้อมูลจะหายเมื่อ restart server
- YOLOv8n-Seg เป็นโมเดลที่ train จาก COCO จึงอาจตรวจจับสินค้าบางประเภทไม่ได้
- ระบบ live ยังไม่มี detector เฉพาะสำหรับ watermark, screenshot, AI-generated และ stock image
- threshold ปัจจุบันเป็นค่า baseline ควร validate กับภาพจริงที่มี label
- API ยังใช้ชื่อไฟล์เป็น product identifier ชั่วคราว
- ยังไม่มี authentication และการเชื่อมต่อกับ `public.ocr`
