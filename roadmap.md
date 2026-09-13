# Roadmap: TTT AI Duplicate Detection Engine

เอกสารนี้สะท้อนสถานะจริงของ repository และลำดับงานถัดไป โดยมุ่งให้ระบบใช้งานได้จริงในขอบเขตแคบและ conservative ก่อน ใช้ YOLOv8n เดิมต่อ และไม่ฝืนตัดสินเมื่อหลักฐานไม่พอ

## สถานะปัจจุบัน

### สิ่งที่ทำงานอยู่แล้ว

- FastAPI pipeline ที่ `/gateway/detectDuplicateProduct`
- Quality gate ก่อนเข้า inference เพื่อตัดภาพเบลอ มืด contrast ต่ำ หรือความละเอียดต่ำ
- YOLOv8n-Seg segmentation และการสร้าง foreground/background โดยมี fallback เป็น full-image embedding เมื่อ segmentation ไม่ผ่าน
- OpenCLIP embedding และ pHash matching
- Scoring ของ live API 4 สถานะ: `UNIQUE`, `REVIEW`, `DUPLICATE`, `INVALID_DATA`
- `SPAM` live logic เริ่มทำงานเมื่อ multipart request มี `product_id`, `seller_id`, `listing_id` และ `category` ครบ และพบ duplicate ของ seller เดิม; request ที่ขาด field จะไม่เข้า SPAM logic
- Phase 1 threshold ใน API: `0.95` สำหรับ duplicate และ `0.70` สำหรับ review
- Detector แบบ conservative สำหรับ screenshot, watermark และ AI artifact
- Evaluation harness สำหรับ Training, Validation และ Testing จาก numeric dataset
- Docker และ in-memory/Supabase reference-store integration แต่ Supabase ต้องตั้งค่า `.env` และรัน schema เองก่อนใช้งาน
- Postman contract ปัจจุบันส่ง `image`, `api-caller` และ `api-key` เท่านั้น และยังมีคำอธิบายเดิมที่ระบุว่าเป็น mock/random result

### หลักฐานการตรวจสอบล่าสุด

- Unit tests ผ่าน 5 เคส และ compile check ผ่าน
- Evaluator รันกับ `dataset-ai-ตรวจสอบรูปภาพซ้ำ.xlsx` ได้ Training 20,000, Validation 2,000 และ Testing 2,000 records
- ผล evaluator เป็นการทดสอบ decision logic จาก precomputed signals ไม่ใช่การวัด YOLO/OpenCLIP จากภาพจริง
- การทดสอบโดยไม่อ่าน `Ground_Truth` ยังพบ `SPAM` จาก rule `BR010` ได้ แต่เป็นการทดสอบ classifier ไม่ใช่หลักฐานว่า live API ตรวจ spam จากภาพจริงได้

### ข้อจำกัดปัจจุบัน

- ใช้ YOLOv8n COCO-pretrained จึงไม่รับประกันการตรวจจับสินค้าทุกชนิด
- ยังไม่มีชุดภาพจริงที่มี Ground Truth สำหรับวัด model pipeline เต็มรูปแบบ
- Detector เป็น heuristic และทำหน้าที่ส่งสัญญาณ `REVIEW` ไม่ใช่การยืนยันเด็ดขาด
- ยังไม่ทำ Stock image detection เพราะไม่มี reference image database
- Live API ยังไม่รับ `product_id`, `seller_id`, `listing_id` หรือ context สำหรับ business rule
- `api-caller` และ `api-key` รับเข้ามาแต่ยังไม่ได้ตรวจ authentication จริง
- `InMemoryReferenceStore` เป็นค่าเริ่มต้น ข้อมูลหายเมื่อ restart และไม่ใช้ nearest-neighbor `top_k` จริง
- ยังไม่มี performance baseline ที่ commit ไว้ใน repository

## Scope สินค้าที่รองรับ

โฟกัสเฉพาะกลุ่มที่ผู้ใช้กำหนด:

- มือถือและแท็บเล็ต
- เสื้อผ้าและแฟชั่น
- กล้อง
- อุปกรณ์เครื่องมือพื้นฐาน
- ความงามและสุขภาพ
- คอมพิวเตอร์และ IT เช่น laptop, monitor, keyboard, mouse, router, speaker และ headphones

ภาพที่ quality gate หรือ detector คัดออกจะเป็น `REVIEW` แต่กรณี segmentation ไม่พบ mask ที่ใช้ได้ ปัจจุบัน extractor จะ fallback ไปใช้ full image embedding และยังไม่ได้บังคับให้เป็น `REVIEW` นี่เป็น gap ที่ต้องแก้/validate ต่อ

## สิ่งที่จะอัปเดต

ลำดับงานใหม่เริ่มจากทำให้ `SPAM` ใช้งานได้ผ่าน live API โดยไม่ให้ client ส่งค่า `Business_Rule` หรือ `Seller_Match` เพื่อบังคับผลลัพธ์เอง จากนั้นจึงทดสอบ integration, performance และ production readiness

## Phase 1 — ทำให้ SPAM ทำงานผ่าน Live API

สถานะ: **ทำแล้วใน prototype; ต้องเพิ่ม authentication และ validate policy ก่อน production**

| งาน | สถานะ | รายละเอียด |
|---|---|---|
| 1.1 กำหนดความหมายของ `BR010` | ทำแล้วสำหรับ prototype | `BR010` ใน live path หมายถึง duplicate ที่ matched reference มี `seller_id` เดียวกับ request และ request context ครบ |
| 1.2 เพิ่ม request context | ทำแล้ว | เพิ่ม `product_id`, `seller_id`, `listing_id` และ `category` เป็น multipart fields; ไม่รับ `Business_Rule` หรือ `Seller_Match` จาก client |
| 1.3 คำนวณ business rule ฝั่ง server | ทำแล้วสำหรับ prototype | `evaluate_spam_decision()` คำนวณ seller match จาก request context กับ stored reference |
| 1.4 เชื่อม live decision flow | ทำแล้วสำหรับ prototype | duplicate ของ seller เดิมจะเปลี่ยนเป็น `SPAM`; image-only หรือ context ไม่ครบใช้ image scoring เดิม |
| 1.5 บันทึก seller/product context | ทำแล้วบางส่วน | in-memory และ Supabase reference รองรับ metadata; candidate scope/authenticated identity ยังต้อง harden ก่อน production |
| 1.6 บันทึกเหตุผลของ SPAM | ทำแล้วบางส่วน | response reason และ Supabase analysis schema/payload รองรับ `business_rule`, `seller_id`, `seller_match` และ `spam_reason` |
| 1.7 ป้องกันการปลอมผลลัพธ์ | ค้างอยู่ | ตรวจ `api-caller`/`api-key` และให้ server เป็นผู้โหลด identity/policy ที่เชื่อถือได้ |
| 1.8 อัปเดต Postman | ทำแล้ว | เพิ่ม metadata ที่จำเป็นและแก้คำอธิบาย duplicate endpoint ให้ตรงกับ live implementation |

เกณฑ์ผ่าน Phase 1:

- request ที่มี context ครบและเป็น duplicate ของ seller เดิมคืน `decision=SPAM`
- request ที่ไม่เข้า policy ไม่ถูกตัดเป็น `SPAM` แม้ภาพจะเหมือนกัน
- request ที่ไม่มี seller/product context ไม่เข้า SPAM logic และยังใช้ image scoring เดิม
- client ไม่สามารถส่ง `Business_Rule=BR010` เพื่อบังคับผลลัพธ์ได้
- มี integration test ครบทั้งเข้าเงื่อนไข, ไม่เข้าเงื่อนไข, context หาย และข้อมูลปลอม

## Phase 2 — Decision และ scoring

สถานะ: **เสร็จบางส่วนและใช้งานได้**

| งาน | สถานะ | รายละเอียด |
|---|---|---|
| 2.1 รองรับ decision classes | เสร็จบางส่วน | live API มี `UNIQUE`, `REVIEW`, `DUPLICATE`, `INVALID_DATA`; evaluator/schema มี `SPAM` แต่ต้องรอ Phase 1 เพื่อใช้จริง |
| 2.2 ใช้ threshold เดียวกันระหว่าง evaluator กับ API | เสร็จ | duplicate `>= 0.95`, review `>= 0.70` |
| 2.3 Quality scores ใน pipeline | เสร็จ | blur, brightness, contrast และ noise อยู่ใน response |
| 2.4 Invalid input แบบ graceful | เสร็จบางส่วน | malformed numeric features ตกเป็น `INVALID_DATA`; live metadata validation ยังต้องเพิ่ม |
| 2.5 ปรับ pHash mapping | ทำแล้วบางส่วน | scoring แยก pHash distance กับ similarity แล้ว และ API ใช้ distance `<= 5`; ยังขาด boundary/integration tests |

## Phase 3 — Product segmentation

สถานะ: **MVP เสร็จแล้ว รอภาพจริงเพื่อ calibrate**

| งาน | สถานะ | รายละเอียด |
|---|---|---|
| 3.1 ใช้ YOLOv8n-Seg ต่อ | เสร็จ | ไม่เปลี่ยนโมเดลใน scope นี้ |
| 3.2 Supported-category gate | เสร็จ | อนุญาตเฉพาะกลุ่มสินค้าเป้าหมาย |
| 3.3 Foreground/background separation | เสร็จบางส่วน | สร้างและ clean mask ได้ แต่ถ้า segmentation ไม่น่าเชื่อถือจะ fallback เป็น full image ไม่ใช่ `REVIEW` โดยอัตโนมัติ |
| 3.4 ทดสอบกับภาพจริง | รอข้อมูล | ต้องมีภาพสินค้าและ Ground Truth จาก Data/QA |
| 3.5 Rotation/flip matching | ภายหลัง | ทำเฉพาะเมื่อมีภาพจริงยืนยันว่าเป็นปัญหาสำคัญ |

## Phase 4 — Image quality และ detectors

สถานะ: **ทำพื้นฐานแล้ว ใช้แบบ soft-review**

| งาน | สถานะ | พฤติกรรม |
|---|---|---|
| Quality gate | เสร็จ | ภาพเสียไม่ต้องผ่าน inference ราคาแพง |
| BR006 Screenshot | เสร็จพื้นฐาน | ตรวจ UI chrome/status bar และ OCR เมื่อพร้อมใช้ |
| BR007 Watermark | เสร็จพื้นฐาน | ตรวจ overlay/text ที่ชัดเจน |
| BR008 AI-generated artifact | เสร็จพื้นฐาน | ตรวจเฉพาะ artifact ที่ชัดเจนและกันภาพพื้นเรียบ false positive |
| BR009 Stock image | ตัดออกจาก scope | ยังไม่มี stock reference database |

หลักการของ detector: flag แล้วคืน `REVIEW` พร้อมเหตุผล ไม่คืน `DUPLICATE` จาก detector เพียงอย่างเดียว

## Phase 5 — Localhost performance validation

เอกสารสำหรับส่งงาน QA: [QA_VALIDATION_TESTING_DATASET_REQUIREMENTS.md](QA_VALIDATION_TESTING_DATASET_REQUIREMENTS.md)

คู่มือการรันและลำดับงาน: [PERFORMANCE_AND_CALIBRATION_GUIDE.md](PERFORMANCE_AND_CALIBRATION_GUIDE.md)

ข้อกำหนด Validation/Calibration: [VALIDATION_CALIBRATION_DATASET_REQUIREMENTS.md](VALIDATION_CALIBRATION_DATASET_REQUIREMENTS.md)

รอบนี้ทดสอบเฉพาะ performance ของ FastAPI บน localhost ไม่ใช้สรุป accuracy และไม่ calibrate model threshold

สถานะปัจจุบัน: **ยังไม่มีผล performance baseline ที่ commit ไว้ใน repository** ลำดับถัดไป:

1. รัน baseline แบบ sequential หลัง warm-up
2. วัด latency และ throughput ตามขนาดภาพ
3. ทดสอบ concurrency 2, 4 และ 8 ตามทรัพยากรเครื่อง
4. ตรวจ error rate, timeout, CPU, RAM และ GPU/VRAM ถ้ามี
5. ระบุ bottleneck ก่อนทำ model optimization รอบถัดไป

สิ่งที่ต้องขอจากทีมอื่น:

- ชุดภาพสำหรับ workload performance แยกตามขนาดไฟล์และ resolution
- ยืนยันเครื่องหรือ environment ที่ใช้เป็น performance baseline
- ยืนยัน timeout และ resource limit ที่ต้องการสำหรับ prototype

## Phase 6 — Production readiness

งานหลัง validation:

- เปลี่ยนจาก filename เป็น `product_id` จริงจาก caller
- ตรวจและเปิดใช้ `SupabaseReferenceStore` พร้อม schema/index บน Supabase จริง (implementation มีแล้ว แต่ยังไม่ใช่หลักฐานว่า environment ถูก provision และทดสอบแล้ว)
- เพิ่ม authentication ตาม contract ที่ยืนยันแล้ว
- เพิ่ม structured logging; model version ถูกบันทึกใน Supabase analysis record แล้ว และ detector reasons อยู่ใน response reason
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

**เป้าหมายของรอบถัดไป:** ทำให้ `SPAM` ตัดสินจาก server-side business context ผ่าน live API ได้อย่างตรวจสอบย้อนหลังและทดสอบซ้ำได้ จากนั้นจึงสร้าง performance baseline ของ FastAPI บน localhost
