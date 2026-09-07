# Roadmap: TTT AI Duplicate Detection Engine

เอกสารนี้สะท้อน scope ล่าสุดที่ตกลงกัน: ทำระบบให้ใช้งานได้จริงในขอบเขตแคบและ conservative ก่อน โดยใช้ YOLOv8n เดิมต่อ ไม่เปลี่ยนโมเดล และคืน `REVIEW` เมื่อหลักฐานไม่พอ

## สถานะปัจจุบัน

เสร็จแล้ว:

- FastAPI pipeline ที่ `/gateway/detectDuplicateProduct`
- Quality gate ก่อนเข้า inference เพื่อตัดภาพเบลอ มืด contrast ต่ำ หรือความละเอียดต่ำ
- YOLOv8n-Seg segmentation gate สำหรับกลุ่มสินค้าเป้าหมาย
- OpenCLIP embedding และ pHash matching
- Scoring 5 สถานะ: `UNIQUE`, `REVIEW`, `DUPLICATE`, `SPAM`, `INVALID_DATA`
- Phase 1 threshold ใน API: `0.95` สำหรับ duplicate และ `0.70` สำหรับ review
- Detector แบบ conservative สำหรับ screenshot, watermark และ AI artifact
- Evaluation harness สำหรับ Training, Validation และ Testing จาก numeric dataset
- Docker และ in-memory/Supabase reference-store integration

ข้อจำกัดที่ยอมรับใน MVP:

- ใช้ YOLOv8n COCO-pretrained จึงไม่รับประกันการตรวจจับสินค้าทุกชนิด
- ยังไม่มีชุดภาพจริงที่มี Ground Truth สำหรับวัด model pipeline เต็มรูปแบบ
- Detector เป็น heuristic และทำหน้าที่ส่งสัญญาณ `REVIEW` ไม่ใช่การยืนยันเด็ดขาด
- ยังไม่ทำ Stock image detection เพราะไม่มี reference image database

## Scope สินค้าที่รองรับ

โฟกัสเฉพาะกลุ่มที่ผู้ใช้กำหนด:

- มือถือและแท็บเล็ต
- เสื้อผ้าและแฟชั่น
- กล้อง
- อุปกรณ์เครื่องมือพื้นฐาน
- ความงามและสุขภาพ
- คอมพิวเตอร์และ IT เช่น laptop, monitor, keyboard, mouse, router, speaker และ headphones

ภาพที่ตรวจไม่พบวัตถุเป้าหมาย วัตถุเล็กเกินไป วัตถุมากเกินไป หรือมีความไม่แน่นอน ให้เป็น `REVIEW` ไม่ฝืนตัดสินเป็น `DUPLICATE`

## Phase 1 — Decision และ scoring

สถานะ: **เสร็จบางส่วนและใช้งานได้**

| งาน | สถานะ | รายละเอียด |
|---|---|---|
| 1.1 รองรับ 5 decision classes | เสร็จ | `UNIQUE`, `REVIEW`, `DUPLICATE`, `SPAM`, `INVALID_DATA` |
| 1.2 ใช้ threshold เดียวกันระหว่าง evaluator กับ API | เสร็จ | duplicate `>= 0.95`, review `>= 0.70` |
| 1.3 Quality scores ใน pipeline | เสร็จ | blur, brightness, contrast และ noise อยู่ใน response |
| 1.4 Invalid input แบบ graceful | เสร็จบางส่วน | malformed numeric features ตกเป็น `INVALID_DATA` |
| 1.5 ปรับ pHash mapping | ถัดไป | แยก pHash distance จากเปอร์เซ็นต์ similarity ให้ชัดเจนและทดสอบ boundary |

## Phase 2 — Product segmentation

สถานะ: **MVP เสร็จแล้ว รอภาพจริงเพื่อ calibrate**

| งาน | สถานะ | รายละเอียด |
|---|---|---|
| 2.1 ใช้ YOLOv8n-Seg ต่อ | เสร็จ | ไม่เปลี่ยนโมเดลใน scope นี้ |
| 2.2 Supported-category gate | เสร็จ | อนุญาตเฉพาะกลุ่มสินค้าเป้าหมาย |
| 2.3 Foreground/background separation | เสร็จ | ถ้า segmentation ไม่น่าเชื่อถือให้ `REVIEW` |
| 2.4 ทดสอบกับภาพจริง | รอข้อมูล | ต้องมีภาพสินค้าและ Ground Truth จาก Data/QA |
| 2.5 Rotation/flip matching | ภายหลัง | ทำเฉพาะเมื่อมีภาพจริงยืนยันว่าเป็นปัญหาสำคัญ |

## Phase 3 — Image quality และ detectors

สถานะ: **ทำพื้นฐานแล้ว ใช้แบบ soft-review**

| งาน | สถานะ | พฤติกรรม |
|---|---|---|
| Quality gate | เสร็จ | ภาพเสียไม่ต้องผ่าน inference ราคาแพง |
| BR006 Screenshot | เสร็จพื้นฐาน | ตรวจ UI chrome/status bar และ OCR เมื่อพร้อมใช้ |
| BR007 Watermark | เสร็จพื้นฐาน | ตรวจ overlay/text ที่ชัดเจน |
| BR008 AI-generated artifact | เสร็จพื้นฐาน | ตรวจเฉพาะ artifact ที่ชัดเจนและกันภาพพื้นเรียบ false positive |
| BR009 Stock image | ตัดออกจาก scope | ยังไม่มี stock reference database |

หลักการของ detector: flag แล้วคืน `REVIEW` พร้อมเหตุผล ไม่คืน `DUPLICATE` จาก detector เพียงอย่างเดียว

## Phase 4 — Localhost performance validation

เอกสารสำหรับส่งงาน QA: [QA_VALIDATION_TESTING_DATASET_REQUIREMENTS.md](QA_VALIDATION_TESTING_DATASET_REQUIREMENTS.md)

คู่มือการรันและลำดับงาน: [PERFORMANCE_AND_CALIBRATION_GUIDE.md](PERFORMANCE_AND_CALIBRATION_GUIDE.md)

ข้อกำหนด Validation/Calibration: [VALIDATION_CALIBRATION_DATASET_REQUIREMENTS.md](VALIDATION_CALIBRATION_DATASET_REQUIREMENTS.md)

รอบนี้ทดสอบเฉพาะ performance ของ FastAPI บน localhost ไม่ใช้สรุป accuracy และไม่ calibrate model threshold

ลำดับถัดไป:

1. รัน baseline แบบ sequential หลัง warm-up
2. วัด latency และ throughput ตามขนาดภาพ
3. ทดสอบ concurrency 2, 4 และ 8 ตามทรัพยากรเครื่อง
4. ตรวจ error rate, timeout, CPU, RAM และ GPU/VRAM ถ้ามี
5. ระบุ bottleneck ก่อนทำ model optimization รอบถัดไป

สิ่งที่ต้องขอจากทีมอื่น:

- ชุดภาพสำหรับ workload performance แยกตามขนาดไฟล์และ resolution
- ยืนยันเครื่องหรือ environment ที่ใช้เป็น performance baseline
- ยืนยัน timeout และ resource limit ที่ต้องการสำหรับ prototype

## Phase 5 — Production readiness

งานหลัง validation:

- เปลี่ยนจาก filename เป็น `product_id` จริงจาก caller
- เปิดใช้ `PgVectorReferenceStore` และตรวจ schema/index
- เพิ่ม authentication ตาม contract ที่ยืนยันแล้ว
- เพิ่ม structured logging, model version และ detector reasons
- เพิ่ม lifecycle สำหรับ update/delete reference image
- ทดสอบ Docker deployment และ load เบื้องต้น

## ลำดับการทำงานตอนนี้

```text
ตอนนี้
       -> รัน localhost FastAPI baseline หลัง warm-up
       -> วัด latency, throughput และ resource usage
       -> ทดสอบ concurrency และ error handling
       -> ระบุ bottleneck ของ pipeline
       -> ค่อยวางแผน model optimization และ production validation
```

**เป้าหมายของรอบถัดไป:** ได้ performance baseline ที่ทำซ้ำได้ของ FastAPI บน localhost พร้อมค่า latency, throughput, concurrency ที่เหมาะสม และ bottleneck ที่ชัดเจน
