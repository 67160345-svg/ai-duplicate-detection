# โครงสร้างโปรเจค Alpha

โปรเจคนี้เป็นบริการ FastAPI สำหรับตรวจจับภาพสินค้าที่ซ้ำกัน โดยประมวลผลภาพด้วย YOLOv8-Seg, OpenCLIP และ pHash จากนั้นใช้ scoring engine ตัดสินผลว่าเป็นภาพซ้ำหรือไม่

## โครงสร้างไดเรกทอรี

```text
alpha/
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── start_server_ngrok.bat
├── README.md
├── PROJECT_STRUCTURE.md
├── PERFORMANCE_AND_CALIBRATION_GUIDE.md
├── QA_VALIDATION_TESTING_DATASET_REQUIREMENTS.md
├── VALIDATION_CALIBRATION_DATASET_REQUIREMENTS.md
├── roadmap.md
│
├── db/
│   └── schema.sql
│
├── modules/
│   ├── detectors.py
│   ├── feature_extractor.py
│   ├── preprocessor.py
│   ├── reference_store.py
│   ├── scoring_engine.py
│   ├── supabase_reference_store.py
│   └── yolov8n-seg.pt
│
├── evaluation/
│   ├── evaluate_dataset.py
│   └── report.json
│
└── tests/
    └── test_quality_gate.py
```

## ไฟล์ระดับ Root

- `main.py` จุดเริ่มต้นของ FastAPI application สร้าง API endpoint รับไฟล์ภาพ ตรวจสอบชนิดและขนาดไฟล์ เรียกใช้ pipeline หลัก และส่งผลลัพธ์กลับเป็น JSON
- `requirements.txt` รายการแพ็กเกจ Python ที่จำเป็นสำหรับ API, computer vision, model inference, Supabase และการประเมินผล
- `Dockerfile` คำสั่งสำหรับสร้าง image ของแอปพลิเคชันเพื่อรันใน Docker
- `docker-compose.yml` การตั้งค่าสำหรับรัน service ด้วย Docker Compose
- `start_server_ngrok.bat` สคริปต์สำหรับเริ่ม FastAPI ใน Windows ปัจจุบันสคริปต์นี้เริ่มเฉพาะ FastAPI ไม่ได้เริ่ม ngrok โดยอัตโนมัติ
- `README.md` ภาพรวม วิธีติดตั้ง การตั้งค่า environment การรัน API และข้อจำกัดของ prototype
- `PROJECT_STRUCTURE.md` เอกสารฉบับนี้ อธิบายโครงสร้างและความรับผิดชอบของแต่ละส่วนในโปรเจค
- `roadmap.md` แผนงานและแนวทางพัฒนาต่อของระบบ

## เอกสารด้านการทดสอบและการปรับเทียบ

- `PERFORMANCE_AND_CALIBRATION_GUIDE.md` แนวทางทดสอบ performance และปรับค่า threshold ของโมเดลกับข้อมูลจริง
- `QA_VALIDATION_TESTING_DATASET_REQUIREMENTS.md` ข้อกำหนดชุดข้อมูลและขั้นตอนสำหรับทดสอบ API, latency, throughput และ resource usage
- `VALIDATION_CALIBRATION_DATASET_REQUIREMENTS.md` ข้อกำหนดชุดข้อมูล Ground Truth สำหรับ validation และ calibration ของระบบตรวจจับภาพ

## โฟลเดอร์ `modules/`

เป็นส่วนประกอบหลักของ business logic และ image-processing pipeline

- `preprocessor.py`
  - โหลดภาพจาก bytes ที่รับจาก multipart upload
  - แปลงภาพเป็น RGB NumPy array
  - คำนวณคะแนน blur, brightness, contrast และ noise
  - ตรวจสอบ quality gate ก่อนส่งภาพเข้าสู่โมเดล

- `feature_extractor.py`
  - โหลด YOLOv8-Seg และ OpenCLIP
  - ตรวจจับและแยก foreground ของสินค้าออกจาก background
  - สร้าง foreground/background embedding
  - สร้างค่า pHash สำหรับเปรียบเทียบภาพ
  - รวม feature ที่ใช้ในขั้นตอน scoring

- `detectors.py`
  - ตรวจจับสัญญาณของ screenshot, watermark และ AI-generated artifact
  - ใช้ heuristic แบบ conservative
  - หากพบสัญญาณที่น่าสงสัย ระบบจะส่งผลเป็น `REVIEW`

- `reference_store.py`
  - กำหนด interface `ReferenceImageStore` สำหรับค้นหาและบันทึกภาพอ้างอิง
  - มี `InMemoryReferenceStore` สำหรับ development, test และ prototype
  - store แบบ memory จะหายเมื่อ process restart และไม่ควรใช้ใน production

- `supabase_reference_store.py`
  - implementation ของ reference store ที่เชื่อมต่อ Supabase
  - เรียก RPC `match_product_images()` เพื่อค้นหา candidate ด้วย vector similarity
  - บันทึก feature และ metadata ลงตาราง `product_reference_images`
  - อัปโหลดไฟล์ไปยัง Supabase Storage

- `scoring_engine.py`
  - คำนวณ cosine similarity และ pHash similarity
  - ตรวจสอบว่า feature ที่รับเข้ามาถูกต้องหรือไม่
  - ประเมิน threshold และตัดสินผล เช่น `DUPLICATE`, `REVIEW`, `UNIQUE`, `SPAM` และ `INVALID_DATA`
  - มี logic สำหรับจำแนกข้อมูล numeric dataset ที่ใช้ในการ evaluation

- `yolov8n-seg.pt`
  - weight ของโมเดล YOLOv8n-Seg ที่ใช้ segmentation
  - `feature_extractor.py` จะโหลดไฟล์นี้เป็นค่าเริ่มต้น
  - สามารถเปลี่ยนตำแหน่งไฟล์ด้วย environment variable `YOLO_SEG_MODEL_PATH`

## โฟลเดอร์ `db/`

- `schema.sql`
  - schema สำหรับ Supabase/PostgreSQL
  - สร้างตารางเก็บภาพอ้างอิงและผลการวิเคราะห์
  - ประกาศฟังก์ชัน RPC `match_product_images()` สำหรับค้นหา candidate ด้วย vector
  - ต้องเปิดใช้ extension ที่เกี่ยวข้อง เช่น `vector` และ `pgcrypto` ตาม README

## โฟลเดอร์ `evaluation/`

- `evaluate_dataset.py`
  - อ่านข้อมูล CSV หรือ Excel ที่มีค่า signal และ Ground Truth อยู่แล้ว
  - ใช้ threshold และกฎจาก `scoring_engine.py` เพื่อจำแนกผล
  - สร้าง metrics สำหรับชุด Training, Validation และ Testing
  - เป็นการประเมิน decision logic จาก precomputed signals ไม่ใช่การรัน YOLO/OpenCLIP กับภาพจริง

- `report.json`
  - ผลลัพธ์การประเมิน dataset ที่บันทึกไว้ในรูป JSON

## โฟลเดอร์ `tests/`

- `test_quality_gate.py`
  - ทดสอบกฎ image quality gate และพฤติกรรมการคัดกรองภาพที่มีคุณภาพไม่เพียงพอ

## ลำดับการทำงานของระบบ

```text
Client uploads image
        │
        ▼
main.py รับและตรวจสอบไฟล์
        │
        ▼
preprocessor.py ตรวจสอบคุณภาพและแปลงภาพ
        │
        ├── ไม่ผ่าน quality gate -> REVIEW
        │
        ▼
feature_extractor.py
YOLOv8-Seg + OpenCLIP + pHash
        │
        ▼
detectors.py ตรวจ screenshot/watermark/AI artifact
        │
        ▼
reference_store ค้นหา candidate images
(InMemory หรือ Supabase)
        │
        ▼
scoring_engine.py เปรียบเทียบ feature และใช้ threshold
        │
        ▼
main.py ส่งผลลัพธ์ API เช่น UNIQUE, REVIEW หรือ DUPLICATE
```

## โหมดจัดเก็บข้อมูล

ระบบเลือก reference store จาก environment variable `REFERENCE_STORE`

```env
REFERENCE_STORE=memory
```

เหมาะสำหรับ local development และการทดสอบ โดยข้อมูลจะอยู่ใน RAM เท่านั้น

```env
REFERENCE_STORE=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_STORAGE_BUCKET=product-images
```

เหมาะสำหรับการเชื่อมต่อฐานข้อมูลและ storage จริง โดยต้องตั้งค่า Supabase ให้ครบก่อนเริ่ม server

## จุดเริ่มต้นที่ควรอ่านเมื่อแก้ไขระบบ

1. เริ่มจาก `main.py` เพื่อดู API contract และลำดับการเรียก pipeline
2. อ่าน `preprocessor.py` หากแก้เรื่องคุณภาพหรือรูปแบบภาพนำเข้า
3. อ่าน `feature_extractor.py` หากแก้ model, segmentation หรือ feature
4. อ่าน `scoring_engine.py` หากแก้ threshold หรือ decision
5. อ่าน `reference_store.py` และ `supabase_reference_store.py` หากแก้การค้นหา/บันทึกข้อมูล
6. รันชุดทดสอบและ compile check หลังแก้ไข

คำสั่ง compile check:

```powershell
.venv\Scripts\python.exe -m compileall -q main.py modules evaluation
```
