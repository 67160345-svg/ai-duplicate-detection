# SPAM Feature Problem Analysis

เอกสารนี้อธิบายปัญหาและขอบเขตของฟีเจอร์ `SPAM` ตาม implementation ปัจจุบัน โดยคงรูปแบบ decision แยก `SPAM` ออกจาก `DUPLICATE` ไว้ตามเดิม เอกสารนี้ไม่ได้เปลี่ยน logic ของระบบ

## 1. ความหมายของ SPAM ในปัจจุบัน

ใน prototype ปัจจุบัน `SPAM` หมายถึง:

> พบภาพซ้ำกับ reference และ reference นั้นเป็นของ seller เดียวกับ request โดย request ต้องมี context ครบถ้วน

เงื่อนไขที่ live API ต้องผ่านทั้งหมด:

- `product_id` ไม่ว่าง
- `seller_id` ไม่ว่าง
- `listing_id` ไม่ว่าง
- `category` ไม่ว่าง
- image scoring ต้องได้ `DUPLICATE`
- reference ที่ match ต้องมี `seller_id`
- `seller_id` ของ request ต้องตรงกับ `seller_id` ของ reference

ระบบไม่ได้รับ `Business_Rule` หรือ `Seller_Match` จาก client โดยตรง แต่คำนวณผลภายในจาก context และ reference ที่ match

## 2. ลำดับการทำงานของ Live API

```text
multipart request
  -> validate image type/size
  -> quality gate
  -> screenshot/watermark/AI-artifact detectors
  -> YOLO/OpenCLIP/pHash feature extraction
  -> reference-store candidate search
  -> image similarity decision
  -> SPAM context gate
  -> save reference and analysis result
  -> return response
```

`SPAM` จะถูกตรวจหลังจาก image similarity ได้ผลก่อน ดังนั้น `SPAM` ไม่ได้เกิดจาก seller context เพียงอย่างเดียว ต้องมีหลักฐานว่า image เป็น duplicate ด้วย

## 3. พฤติกรรมตาม input

| Input | ผลที่เป็นไปได้ | เข้า SPAM หรือไม่ |
|---|---|---|
| ส่ง `image` อย่างเดียว | `UNIQUE`, `REVIEW` หรือ `DUPLICATE` | ไม่เข้า |
| ส่ง context มาไม่ครบ | image decision เดิม | ไม่เข้า |
| ส่ง context ครบ แต่ภาพไม่ซ้ำ | `UNIQUE` หรือ `REVIEW` | ไม่เข้า |
| ส่ง context ครบ ภาพซ้ำกับ seller อื่น | `DUPLICATE` | ไม่เข้า |
| ส่ง context ครบ ภาพซ้ำกับ seller เดิม | `SPAM` | เข้า |

ตัวอย่างการใช้งาน:

```text
Request 1:
image + product_id=P1 + seller_id=S1 + listing_id=L1 + category=camera
=> UNIQUE และบันทึก reference ของ S1

Request 2:
image ซ้ำ + product_id=P2 + seller_id=S1 + listing_id=L2 + category=camera
=> DUPLICATE จาก image scoring แล้วเปลี่ยนเป็น SPAM

Request 3:
image ซ้ำ + ไม่มี context
=> DUPLICATE แต่ไม่เป็น SPAM
```

## 4. ความหมายของ SPAM ใน dataset

ไฟล์ `dataset-ai-ตรวจสอบรูปภาพซ้ำ.xlsx` เป็น dataset ที่มี precomputed signals และเฉลย ไม่ใช่ raw image dataset

แถว `SPAM` ใน dataset มีรูปแบบหลักดังนี้:

- `Business_Rule = BR010`
- `Scenario_Type = SPAM`
- `Ground_Truth = SPAM`
- `Seller_Match = Yes`
- `Embedding_Similarity` อยู่ในช่วงสูงประมาณ `0.97-1.00`

Evaluator เดิมตัดสิน `SPAM` จาก `BR010` และ `Seller_Match=Yes` เป็นหลัก ส่วน live API ปัจจุบันใช้เงื่อนไขเข้มกว่า โดยต้องมี request context ครบและตรวจ seller match จาก reference จริง

ดังนั้นผล `SPAM` ใน dataset และผล `SPAM` ใน live API มีแนวคิดใกล้กัน แต่ไม่ได้ใช้ input contract เดียวกันทั้งหมด

## 5. สิ่งที่ SPAM ยังไม่ได้ตรวจ

ปัจจุบัน `SPAM` ยังไม่ใช่การตรวจพฤติกรรม spam แบบเต็มรูปแบบ ระบบยังไม่ได้ใช้:

- จำนวนครั้งที่ seller อัปโหลดภาพซ้ำ
- ช่วงเวลาระหว่างการอัปโหลด
- จำนวน listing ที่ใช้ภาพเดียวกัน
- IP address หรือ device fingerprint
- ประวัติการถูก block หรือ violation
- rate limit หรือ quota
- สถานะบัญชี seller
- การยืนยันว่า listing เดิมยัง active อยู่

ด้วยเหตุนี้ คำว่า `SPAM` ในระบบปัจจุบันควรอ่านว่า **same-seller duplicate ตาม business rule prototype** ไม่ใช่การยืนยันว่าผู้ใช้มีพฤติกรรม spam จริง

## 6. เหตุผลที่แยก SPAM ออกจาก DUPLICATE

การแยก decision มีประโยชน์หาก downstream ต้องดำเนินการต่างกัน:

- `DUPLICATE`: แจ้งเตือน, ส่งตรวจสอบ หรือแสดง reference ที่ match
- `SPAM`: block, reject, จำกัด quota หรือสร้าง audit event

อย่างไรก็ตาม ใน implementation ปัจจุบัน `SPAM` เป็น subset ของ `DUPLICATE` เพราะต้องได้ duplicate ก่อนเสมอ หากระบบไม่มี action ที่ต่างกันจริง การเก็บ `SPAM` แยกอาจทำให้ผู้ใช้ตีความเกินข้อมูลที่ระบบมี

## 7. ข้อจำกัดทางเทคนิคที่ต้องรับรู้

### 7.1 Authentication ยังไม่บังคับใช้

`api-caller` และ `api-key` รับเข้ามาใน endpoint แต่ยังไม่ได้ตรวจสอบจริง ดังนั้น `seller_id` ที่ส่งจาก client ยังไม่ควรถือเป็น identity ที่เชื่อถือได้ใน production

### 7.2 Candidate scope ยังไม่ได้ส่งจาก endpoint

reference store รองรับ `scope` เช่น seller/category แต่ live endpoint ปัจจุบันเรียก `find_candidates()` โดยไม่ได้ส่ง scope ดังนั้นการค้นหา candidate อาจพิจารณา reference หลาย seller แล้วค่อยตรวจ seller match หลังจากเลือก matched record

ก่อน production ควรยืนยันว่าจะ:

- filter candidate ตาม seller ก่อนค้นหา หรือ
- ค้นหาทุก seller แล้วใช้ policy ที่ชัดเจนในการเลือก candidate

### 7.3 `repetition_rate` ของ SPAM

เมื่อ decision ถูกเปลี่ยนจาก `DUPLICATE` เป็น `SPAM`, `is_repetition` เป็น `true` แต่ `repetition_rate` ปัจจุบันถูกคำนวณเฉพาะ decision `DUPLICATE` หรือ `REVIEW` จึงอาจคืนค่า `0.0` ในกรณี `SPAM` แม้ `foreground_similarity` จะสูง

นี่เป็นความไม่สอดคล้องของ response contract ที่ควรระบุไว้ก่อนให้ client นำค่าไปใช้

### 7.4 Metadata ยังมีความเสี่ยงด้านความถูกต้อง

ระบบบันทึก `product_id`, `seller_id`, `listing_id` และ `category` จาก multipart input โดยตรง หากไม่มี authentication หรือการตรวจสอบกับฐานข้อมูล อาจเกิดการส่ง identity ปลอมเพื่อหลีกเลี่ยงหรือทำให้เข้า SPAM policy

## 8. สิ่งที่ยืนยันแล้ว

- image-only request ไม่เข้า SPAM logic
- context ไม่ครบไม่เข้า SPAM logic
- complete context + same-seller duplicate ได้ `SPAM`
- complete context + duplicate จาก seller อื่นยังเป็น `DUPLICATE`
- client ไม่ได้ส่ง `Business_Rule` หรือ `Seller_Match` เพื่อบังคับผลลัพธ์
- unit tests และ integration smoke test สำหรับเงื่อนไขหลักผ่านแล้ว

## 9. ขอบเขตของเอกสารนี้

เอกสารนี้เป็นปัญหาและข้อจำกัดของ implementation ปัจจุบัน ไม่ใช่ข้อกำหนด production final และไม่ใช่หลักฐานว่าโมเดลมีความแม่นยำในการตรวจ spam จากภาพจริง

การเปลี่ยน policy เช่น ใช้จำนวนครั้ง, time window, listing status หรือ seller account history ต้องตกลง business rule ใหม่และเพิ่มข้อมูลที่ระบบสามารถตรวจสอบได้ก่อนแก้ decision logic
