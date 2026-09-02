# Roadmap: TTT AI Duplicate Detection Engine

**อ้างอิงจาก:** โค้ดปัจจุบัน (หลัง quick fixes รอบล่าสุด) + dataset ที่เพิ่งได้รับ (20K/2K/2K records +
Business Rules 12 ข้อ) + หน้าจอ frontend ที่ทีมหลักทำ + postman collection เดิม

**หลักการจัดลำดับ:** ทำสิ่งที่ **ไม่ต้องรอใคร** ก่อน (อยู่ในมือคุณคนเดียว) แล้วค่อยขยับไปงานที่ **ต้องรอคำตอบ
จากทีมอื่น** — ไม่ปล่อยให้ blocked items ทำให้งานทั้งหมดหยุดรอ

---

## Phase 0 — วัดของจริงก่อนแก้อะไร (ทำได้ทันที ไม่มี blocker)

> เป้าหมาย: รู้ตัวเลข precision/recall จริงของ logic ปัจจุบัน ก่อนเริ่มแก้อะไรทั้งนั้น จะได้แก้แบบมีข้อมูลรองรับ
> ไม่ใช่เดา

| งาน | รายละเอียด | Output |
|---|---|---|
| 0.1 เขียน evaluation harness | ดึง decision logic จาก `scoring_engine.py` มาทดสอบกับ `Embedding_Similarity`/`Perceptual_Hash` ที่มีอยู่แล้วใน dataset (ไม่ต้องมีภาพจริง เพราะ dataset mock ตัวเลขไว้แล้ว) | Precision/Recall/F1 ต่อ scenario type บน Training set |
| 0.2 รันกับ Validation set | ปรับ threshold คร่าวๆ จาก Business Rules classification table (0.99/0.95/0.85/0.70) แล้ววัดผลกับ Validation set | ตัวเลขเทียบก่อน/หลังปรับ threshold |
| 0.3 รันกับ Testing set (ครั้งเดียว) | ยืนยันผลสุดท้ายบน held-out set ที่ไม่เคยใช้ tune เลย | ตัวเลข final ที่เชื่อถือได้ ไม่ overfit |
| 0.4 ขอ UAT dataset ที่ขาดหาย | Data Dictionary อ้างถึง Worksheet 6 (UAT) แต่ไฟล์ที่ได้มามีแค่ 01-05 | ไฟล์ครบ หรือคำยืนยันว่ายังไม่ทำ |

**ข้อจำกัดที่ต้องรู้ล่วงหน้า:** Phase นี้ทดสอบได้แค่ **ชั้น decision/scoring logic** เท่านั้น เพราะ dataset ไม่มีภาพจริงแนบมา — จะยังไม่รู้ว่า `feature_extractor.py` (YOLO/CLIP/pHash ตัวจริง) สกัด feature จากภาพแม่นแค่ไหน

---

## Phase 1 — ยกเครื่อง scoring logic ให้ตรงสเปกธุรกิจ (ทำได้เอง ไม่ต้องรอใคร)

> เป้าหมาย: จาก binary (DUPLICATE/UNIQUE) ไปเป็น 5 คลาสตามที่ dataset/business rules ต้องการจริง

| งาน | อ้างอิง Business Rule | ความซับซ้อน |
|---|---|---|
| 1.1 เปลี่ยน output เป็น 5 คลาส (UNIQUE/REVIEW/DUPLICATE/SPAM/INVALID_DATA) | ทุกข้อ | 🔴 ใหญ่ — กระทบ response contract ทั้งหมด (`GatewayDetectionResponse`) |
| 1.2 ปรับ threshold ตาม classification table (0.99/0.95/0.85/0.70) แทน `fg_threshold=0.85` เส้นเดียว | BR001, BR002 | 🟡 กลาง |
| 1.3 Map pHash distance ให้ตรงกับ % similarity ในตาราง (ตอนนี้ผูกกับ `phash_distance_threshold=5` ซึ่งเป็นคนละหน่วยกับ % ในตาราง) | BR001 | 🟡 กลาง |
| 1.4 Wire `calculate_quality_scores()` เข้า pipeline จริง (ตอนนี้เป็น dead code ใน `preprocessor.py` ไม่เคยถูกเรียก) | รองรับ Blur/Brightness/Contrast/Noise ใน response | 🟢 เล็ก |
| 1.5 จัดการ input ที่ผิดปกติแบบ graceful (null similarity, missing field) ให้ตกเป็น `INVALID_DATA` แทนที่จะ error/crash | Scenario `NEGATIVE` ใน dataset | 🟡 กลาง |

---

## Phase 2 — แก้ core algorithm (segmentation) แล้ววัดผลซ้ำ

> เป้าหมาย: แก้ปัญหาที่กระทบมากที่สุด — YOLO-COCO ตรวจจับสินค้าจริงไม่ได้ ทำให้ BR011 (พื้นหลังคล้ายแต่คนละ
> สินค้า) ใช้งานไม่ได้จริง

| งาน | รายละเอียด |
|---|---|
| 2.1 หาทางเลือกแทน YOLOv8n-seg (COCO-pretrained) | ตัวเลือก: fine-tune บน product category จริง / ใช้ class-agnostic segmentation (เช่น SAM, rembg) แทนการพึ่ง fixed class list |
| 2.2 Implement + integrate เข้า `feature_extractor.py` | แทนที่ `segment_foreground_background()` |
| 2.3 รัน Phase 0 evaluation ซ้ำ (แต่คราวนี้ต้องมีภาพจริงแล้ว ไม่ใช่แค่ตัวเลข mock) | ต้องมีชุดภาพจริงมาทดสอบ ไม่ใช่แค่ numeric dataset — เป็นงานเพิ่มที่ยังไม่มีคำตอบว่าจะได้ภาพจริงจากไหน (ดู Phase 4) |
| 2.4 พิจารณา rotation/flip-invariant matching เพิ่มเติม | BR003 (rotate), BR005 (mirror) — pHash ปกติไม่รองรับ อาจต้องพึ่ง CLIP embedding ล้วนๆ หรือเทคนิคเสริม |

---

## Phase 3 — Detector ใหม่ที่ยังไม่มีเลยในระบบ (ขอบเขตงานใหญ่ ต้องตัดสินใจลำดับความสำคัญ)

> ยังไม่มีโค้ดส่วนนี้แม้แต่บรรทัดเดียว — ต้องคุยกับทีม/หัวหน้าก่อนว่าจะทำครบทุกตัวหรือเลือกทำบางตัวก่อน

| Business Rule | Detector ที่ต้องสร้างใหม่ | หมายเหตุ |
|---|---|---|
| BR006 Screenshot detection | ตรวจจับ status bar/UI chrome ในภาพ | อาจใช้ OCR + edge detection ง่ายๆ ก่อน ไม่ต้องเทรนโมเดลใหม่ |
| BR007 Watermark detection | ตรวจจับ logo/text overlay | มีโมเดล open-source สำเร็จรูปพอใช้ได้ |
| BR008 AI-generated image detection | ตรวจจับ artifact จากภาพที่ AI สร้าง | งานยากสุดในกลุ่มนี้ ต้องใช้โมเดลเฉพาะทาง เทคโนโลยียังเปลี่ยนเร็ว |
| BR009 Stock image detection | เทียบกับ embedding cluster ของภาพ stock ที่รู้จัก | ต้องมี database ภาพ stock อ้างอิงก่อน (ยังไม่มี) |

**ข้อเสนอ:** ทำ Phase 0-2 ให้เสร็จก่อน แล้วค่อยกลับมาดูว่า Phase 3 คุ้มทำเองทั้งหมดไหม หรือควรใช้ 3rd-party API สำเร็จรูปสำหรับบางตัว (เช่น watermark/AI-image detection มักมี API สำเร็จรูปที่แม่นกว่าทำเองในเวลาจำกัด)

---

## Phase 4 — งานที่ยัง Blocked รอคำตอบจากทีมอื่น

> เดินหน้า Phase 0-2 ไปพร้อมกันได้ระหว่างรอคำตอบพวกนี้ — แต่ต้องมีคำตอบก่อนเริ่ม Phase 4 จริง

| # | ต้องคุยกับใคร | เรื่อง |
|---|---|---|
| 4.1 | ทีม gateway/backend | Auth (`api-caller`/`api-key`/`Referer`) — service นี้ validate เองหรือ gateway ทำให้แล้ว |
| 4.2 | ทีม gateway/backend | Logging ลง `public.ocr` — ต้องทำเองไหม |
| 4.3 | ทีม product | Identity ของภาพ — จะส่ง `product_id`/`seller_id` มาด้วยไหม (จำเป็นสำหรับ BR010, BR012) |
| 4.4 | ทีม product | Scope การเทียบ (ทั้งแพลตฟอร์ม/เฉพาะร้าน) ที่ต้องเลือกได้ต่อ request ตาม UI ที่เห็น |
| 4.5 | ทีม data/QA | ขอชุดภาพจริงสำหรับทดสอบ segmentation (Phase 2.3) — dataset ที่มีตอนนี้เป็นตัวเลข mock ไม่มีไฟล์ภาพ |
| 4.6 | ทีม data/QA | UAT dataset ที่ขาดหายไป (Phase 0.4) |
| 4.7 | ทีม frontend | Finalize response schema ใหม่ (listing ID ต้นฉบับ, timestamp, risk tag, scope selector) — postman เดิม outdated ไปแล้ว |
| 4.8 | ทีม backend/DBA | Postgres instance สำหรับ `product_reference_images` — เดียวกับ `public.ocr` หรือแยก |

---

## Phase 5 — Infra/Deployment (ส่วนใหญ่เป็นหน้าที่ลูกทีม/ทีม API แล้ว)

| งาน | สถานะ |
|---|---|
| Implement `PgVectorReferenceStore` ตาม `reference_store.py` + `schema.sql` | รอลูกทีม |
| Deploy จริง (แทนการรันผ่าน `.bat` + ngrok ที่เป็นแค่ demo ชั่วคราว) | รอทีม API |
| Auth middleware (ถ้าคำตอบ 4.1 ออกมาว่าต้องทำเอง) | รอคำตอบก่อน |

---

## ลำดับแนะนำโดยสรุป

```
ตอนนี้ → เริ่ม Phase 0 (evaluation harness) ทันที ไม่ต้องรอใคร
       ↘ พร้อมกันนั้น ส่งคำถาม Phase 4.1-4.8 ไปให้ทีมที่เกี่ยวข้องเลย (ยิ่งส่งเร็วยิ่งได้คำตอบเร็ว)

หลัง Phase 0 มีตัวเลขจริง → Phase 1 (ปรับ scoring logic เป็น 5 คลาส + threshold ใหม่)

ขนานกับ Phase 1 → เริ่ม Phase 2.1 (คิดหาทางเลือกแทน YOLO-COCO) ได้เลย เพราะไม่ต้องรอ Phase 1 เสร็จก่อน
                   แต่ Phase 2.3 (วัดผลกับภาพจริง) ต้องรอ Phase 4.5 ก่อน

Phase 3 (detector ใหม่) → รอจนกว่า Phase 0-2 เสถียรก่อน ค่อยประเมินว่าคุ้มทำเองหรือใช้ 3rd-party

Phase 5 → ดำเนินคู่ขนานได้ตลอดเวลาโดยลูกทีม ไม่ต้องรอ Phase 1-3 เสร็จ
```

**จุดที่ควรเริ่มวันนี้เลยถ้าต้องเลือกอย่างเดียว:** Phase 0.1 (evaluation harness) — ใช้เวลาไม่นาน ไม่มี
blocker ใดๆ และให้ข้อมูลที่จำเป็นสำหรับตัดสินใจใน Phase 1-2 ทั้งหมด
