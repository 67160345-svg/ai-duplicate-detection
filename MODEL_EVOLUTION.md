# Model Evolution: Beta1 to Current Version

เอกสารนี้อธิบายความแตกต่างระหว่างระบบรุ่น Beta1 กับเวอร์ชันปัจจุบัน และแนวทางต่อยอดตาม Roadmap

## ภาพรวม

เวอร์ชันปัจจุบันพัฒนาจาก Beta1 ซึ่งเป็นระบบต้นแบบที่เน้นการทดสอบ pipeline หลัก ได้แก่ YOLOv8-Seg, OpenCLIP, pHash และ scoring แบบ `DUPLICATE` / `UNIQUE`

ปัจจุบันระบบยกระดับจาก demo แบบ in-memory ไปเป็น service ที่รองรับการเก็บข้อมูลถาวรด้วย Supabase, vector search, file upload และการเก็บผลวิเคราะห์แยกจากข้อมูลอ้างอิง

## ความแตกต่างจาก Beta1

| หัวข้อ | Beta1 | เวอร์ชันปัจจุบัน |
|---|---|---|
| การรับภาพ | รับไฟล์ภาพ แต่ยังใช้ชื่อไฟล์เป็น reference หลัก | รับ `multipart/form-data` และอัปโหลดไฟล์จริงเข้า Supabase Storage |
| การเก็บข้อมูล | เก็บใน memory ข้อมูลหายเมื่อ restart | เก็บ reference และผลวิเคราะห์ใน Supabase |
| Vector search | เปรียบเทียบแบบ linear scan | ค้นหา candidate ด้วย Supabase pgvector และ RPC |
| ตารางข้อมูล | ยังไม่มีผลวิเคราะห์แยก | มี `product_reference_images` และ `duplicate_analysis_results` |
| File storage | ยังไม่ได้เก็บไฟล์จริง | เก็บไฟล์ใน bucket `product-images` |
| Model path | อาจขึ้นกับ working directory | ใช้ path จากตำแหน่งไฟล์และปรับผ่าน environment ได้ |
| YOLO integration | เรียก model โดยตรง | มี `_predict_segmentation()` พร้อมตั้งค่า confidence/IoU |
| Decision classes | `DUPLICATE`, `UNIQUE` | เพิ่ม `REVIEW`, `SPAM`, `INVALID_DATA` |
| Threshold | ใช้ threshold เดียวเป็นหลัก | มี threshold table สำหรับ exact, near duplicate และ review |
| Quality scores | มีฟังก์ชันแต่ยังไม่อยู่ใน pipeline | ส่ง blur, brightness, contrast และ noise ใน response |
| Error handling | อาจเกิด error จาก input หรือ feature ที่ผิดรูปแบบ | ตรวจ input และตอบข้อความ error แบบ generic |
| Async processing | งาน inference อาจ block event loop | ใช้ threadpool กับ inference และ database calls |
| Evaluation | ยังไม่มี evaluation harness | มี harness สำหรับ Training, Validation และ Testing |
| OpenAPI upload | อาจแสดง image เป็น string | ปรับ schema ให้ Swagger แสดงเป็น file upload |

## สิ่งที่ยังไม่เสร็จสมบูรณ์

- `SPAM` ยังไม่ได้ตรวจจากพฤติกรรม upload จริงใน live endpoint
- ยังไม่มี detector เฉพาะสำหรับ watermark, screenshot, AI-generated image และ stock image
- API ยังใช้ชื่อไฟล์เป็น `product_id` ชั่วคราว
- YOLOv8n-Seg ยังเป็น COCO-pretrained จึงอาจตรวจจับสินค้าบางประเภทไม่ได้
- Evaluation ปัจจุบันใช้ numeric signals และ flags จาก dataset ไม่ใช่การวัดจากภาพจริง
- ยังไม่มี authentication
- ยังไม่มีการเชื่อมต่อกับ `public.ocr`
- ยังไม่มี lifecycle สำหรับ update/delete reference image

## แผนต่อยอดตาม Roadmap

### Phase 2.3: Evaluation ด้วยภาพจริง

เมื่อได้รับภาพจริงจากทีม Data/QA ให้ทดสอบ flow ต่อไปนี้:

```text
ภาพจริง
-> YOLOv8-Seg
-> OpenCLIP + pHash
-> scoring engine
-> เปรียบเทียบกับ Ground Truth
```

เป้าหมายคือวัดความแม่นยำของ model pipeline จริง ไม่ใช่เฉพาะ decision logic จากข้อมูลจำลอง

### Phase 2.4: Rotation และ Flip-invariant Matching

เพิ่มความทนทานต่อ:

- หมุนภาพ 90, 180 และ 270 องศา
- กลับภาพแนวนอน
- กลับภาพแนวตั้ง

ควรใช้ CLIP embedding เป็นสัญญาณหลักเพิ่มเติม เพราะ pHash ปกติไม่ทนต่อการหมุนและการกลับภาพ

### Phase 3: Detector เฉพาะทาง

เพิ่ม detector ตามลำดับความเหมาะสม:

1. Screenshot detection
2. Watermark detection
3. Stock image detection
4. AI-generated image detection

Screenshot และ watermark น่าจะเริ่มได้ง่ายกว่า AI-generated detection ซึ่งต้องใช้ model หรือบริการเฉพาะทาง

### Phase 4: Finalize Contract และ Business Integration

ต้องยืนยันกับทีมที่เกี่ยวข้องเรื่อง:

- `listing_id`
- `seller_id`
- comparison scope
- risk tag
- timestamp ของภาพที่ match
- upload count สำหรับตรวจ spam
- authentication
- logging contract
- lifecycle ของ reference image

### Phase 5: Production Readiness

งานที่ควรทำก่อน production:

- เปลี่ยน temporary product ID เป็น ID จริงจาก caller
- เพิ่ม authentication และกำหนด Supabase access policy
- เพิ่ม update/delete reference image
- เพิ่ม monitoring และ structured logging
- ใช้ database persistence แทน in-memory
- deploy FastAPI บน server ที่เหมาะกับ YOLO/CLIP
- เปลี่ยนจาก ngrok ไปเป็น production deployment

## ข้อควรระวังเรื่องผล Evaluation

ผล Phase 1 ที่ได้ accuracy และ macro-F1 เท่ากับ `1.0` ทุก split ต้องตีความอย่างระมัดระวัง เพราะ dataset มี Business Rule และ detector flags ที่ถูกใช้เป็น input ในการจำแนกอยู่แล้ว

ผลดังกล่าวยืนยันว่า rule mapping ทำงานตรงกับ dataset แต่ยังไม่ใช่หลักฐานว่า YOLO หรือ OpenCLIP มีความแม่นยำ `100%` กับภาพจริง

## สรุป

เวอร์ชันปัจจุบันพร้อมสำหรับการทดสอบ service และการเชื่อม Supabase ในระดับ prototype ที่สมบูรณ์กว่า Beta1 โดยมี persistence, vector search, file storage, quality scores และ evaluation harness

ขั้นถัดไปที่สำคัญที่สุดคือขอชุดภาพจริงเพื่อทำ Phase 2.3 และยืนยัน response contract กับทีม frontend/backend ก่อนขยายไปยัง detector เฉพาะทางและ production deployment
