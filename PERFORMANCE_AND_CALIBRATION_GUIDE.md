# Performance และ Validation Guide สำหรับ Prototype

คู่มือนี้เป็นลำดับการทดสอบที่ QA ใช้ได้ตั้งแต่เริ่ม server จนส่ง report กลับมาให้ calibrate ระบบ

หลักการคือ **ต้องทำ Performance test ก่อน** แล้วจึงทำ Validation test ด้วย Ground Truth เพื่อไม่ให้การปรับโมเดลปนกับปัญหา server overload หรือ timeout

---

## 1. ภาพรวมลำดับงาน

```text
เตรียม environment
  -> Start FastAPI แบบ local หรือ Docker
  -> Smoke test endpoint
  -> Stage A: Performance test
  -> ตรวจ latency/throughput/error/resource
  -> ล็อก build และ environment
  -> Stage B: Functional validation test
  -> ส่ง report + raw response + Ground Truth
  -> Calibrate บน Calibration set
  -> ตรวจซ้ำบน Validation set
  -> Final Testing set รันครั้งเดียวหลังล็อกค่า
```

### Stage A: Performance

วัตถุประสงค์คือรู้ว่า server รับ workload ได้แค่ไหน ไม่ตัดสินว่าโมเดลทายถูกหรือผิด

วัด:

- latency ต่อ request: average, p50, p95, p99, max
- throughput: requests/second
- concurrency ที่เหมาะสม
- timeout และ HTTP 4xx/5xx
- CPU, RAM, GPU/VRAM
- ความแตกต่างระหว่างภาพเล็ก/กลาง/ใหญ่ และภาพที่ quality gate ตัดเร็ว

### Stage B: Functional validation

ทำหลังจาก Stage A ผ่านและใช้ build เดียวกัน

วัตถุประสงค์คือส่งหลักฐานให้ทีมพัฒนา calibrate:

- YOLO foreground/background
- OpenCLIP foreground similarity
- pHash distance/similarity
- quality-gate threshold
- screenshot/watermark/AI-artifact detector
- decision threshold `DUPLICATE` / `REVIEW`

---

## 2. สิ่งที่ต้องล็อกก่อนทดสอบ

QA ต้องบันทึกใน `environment.txt` และ `README.txt`:

- commit/build ที่ใช้
- วันที่และเวลาเริ่มทดสอบ
- OS, CPU, logical cores, RAM
- GPU, VRAM, CUDA ถ้ามี
- Python และ package versions
- รูปแบบการรัน: `local`, `docker-localhost` หรือ `docker-lan`
- `REFERENCE_STORE`
- `YOLO_SEG_CONFIDENCE`
- `YOLO_SEG_IOU`
- `YOLO_SEG_MASK_CONFIDENCE`
- `YOLO_SEG_MAX_OBJECTS`
- จำนวน Uvicorn workers
- port และ URL ที่ client ใช้

ระหว่าง Performance test ห้ามเปลี่ยน model, threshold, environment หรือจำนวน workers

---

## 3. วิธีรันแบบ Local บนเครื่อง Server

เหมาะสำหรับ baseline ที่เร็วที่สุด และใช้ตรวจว่าโค้ดทำงานบนเครื่องเจ้าของ server

### 3.1 ติดตั้ง dependency

เปิด PowerShell ที่ root project:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3.2 ตั้งค่า memory store

```powershell
$env:REFERENCE_STORE="memory"
$env:YOLO_SEG_CONFIDENCE="0.25"
$env:YOLO_SEG_IOU="0.7"
$env:YOLO_SEG_MASK_CONFIDENCE="0.35"
$env:YOLO_SEG_MAX_OBJECTS="3"
```

### 3.3 Start server สำหรับ test

ใช้ `127.0.0.1` เมื่อทดสอบจากเครื่องเดียวกัน:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

ห้ามใช้ `--reload` ใน Performance test เพราะ file watcher ทำให้ผล latency ไม่นิ่ง

### 3.4 Smoke test

เปิด browser:

```text
http://127.0.0.1:8000/docs
```

หรือทดสอบ endpoint จาก PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/gateway/detectDuplicateProduct" -H "api-caller: qa" -H "api-key: prototype-test" -F "image=@C:\test-images\sample.jpg;type=image/jpeg"
```

ต้องได้ response JSON และ server ต้องไม่หยุดทำงาน

### 3.5 ข้อจำกัดของ local mode

ถ้า bind `127.0.0.1` เครื่องอื่นจะเข้าไม่ได้ เหมาะสำหรับ server owner เท่านั้น

---

## 4. วิธีรันด้วย Docker บนเครื่อง Server

เหมาะสำหรับทำ environment ให้ QA ใช้ซ้ำได้และลดความต่างของ Python environment

### 4.1 เตรียม `.env`

ใช้ `REFERENCE_STORE=memory` สำหรับรอบ performance/validation prototype:

```env
REFERENCE_STORE=memory
YOLO_SEG_CONFIDENCE=0.25
YOLO_SEG_IOU=0.7
YOLO_SEG_MASK_CONFIDENCE=0.35
YOLO_SEG_MAX_OBJECTS=3
```

ไม่ต้องใส่ Supabase key ถ้ายังไม่ทดสอบ persistence

### 4.2 Build และ start

```powershell
docker compose build --no-cache
docker compose up -d
```

ตรวจสถานะ:

```powershell
docker compose ps
docker compose logs -f api
```

ทดสอบจาก server:

```powershell
curl.exe http://127.0.0.1:8000/docs
```

หยุดระบบ:

```powershell
docker compose down
```

### 4.3 ข้อควรระวัง Docker

- ใช้ image เดียวกันตลอดรอบทดสอบ
- ห้าม `docker compose build` ใหม่ระหว่าง batch เดียวกัน
- `restart: unless-stopped` อาจทำให้ container restart หลัง crash; QA ต้องบันทึกเหตุการณ์นี้ใน report
- ตรวจ `docker compose logs` ทุกครั้งหลัง stress test
- Dockerfile ปัจจุบันเป็น CPU-oriented; อย่าเปรียบเทียบตัวเลข Docker CPU กับ native GPU โดยตรง

---

## 5. ให้เครื่องอื่นเข้าทดสอบผ่าน LAN

วิธีนี้เหมาะเมื่อ QA อยู่ network เดียวกับเครื่องที่รัน server

### 5.1 Bind service ให้รับจาก network

Local Python:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Docker compose มี mapping `8000:8000` และ container bind `0.0.0.0` อยู่แล้ว

### 5.2 หา IP เครื่อง server

```powershell
ipconfig
```

สมมติ IPv4 เป็น `192.168.1.50` ให้ QA เปิด:

```text
http://192.168.1.50:8000/docs
```

หรือใช้ endpoint:

```text
http://192.168.1.50:8000/gateway/detectDuplicateProduct
```

### 5.3 เปิด Windows Firewall เฉพาะเครือข่ายที่อนุญาต

PowerShell แบบ Administrator บนเครื่อง server:

```powershell
New-NetFirewallRule -DisplayName "Alpha FastAPI QA 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
```

หลังทดสอบเสร็จให้ปิด rule:

```powershell
Remove-NetFirewallRule -DisplayName "Alpha FastAPI QA 8000"
```

QA ต้องทดสอบจากเครื่อง client:

```powershell
Test-NetConnection 192.168.1.50 -Port 8000
```

### 5.4 ข้อจำกัด LAN

- client และ server ต้องอยู่ network เดียวกัน
- ห้ามใช้ IP นี้เป็น public internet endpoint
- service ปัจจุบันยังไม่มี authentication ที่ควรใช้กับ public exposure
- ใช้ API key/header สำหรับการจำลอง test ได้ แต่ไม่ใช่ security จริง

---

## 6. ให้คนนอก network เข้ามาทดสอบ

สำหรับ prototype ให้เลือกอย่างใดอย่างหนึ่ง:

### ทางเลือก A: VPN/องค์กร

แนะนำที่สุดสำหรับข้อมูลภาพจริง:

1. ให้ QA เชื่อม VPN เดียวกับ server
2. ใช้ private IP หรือ internal DNS
3. เปิด firewall เฉพาะ VPN subnet
4. ส่ง URL เช่น `http://10.x.x.x:8000/docs`

### ทางเลือก B: Tunnel ชั่วคราว

ใช้เมื่อ QA อยู่นอก network และต้องทดสอบเร็ว เช่น tunnel service ที่องค์กรอนุมัติ

หลักการ:

1. ให้ FastAPI/Docker ทำงานที่ `127.0.0.1:8000` หรือ port local
2. เปิด tunnel ไปยัง port 8000
3. ส่ง HTTPS URL ให้ QA
4. จำกัดเวลาและปิด tunnel หลังจบ test

ห้ามนำ service prototype ที่ไม่มี authentication ไปเปิด public แบบถาวร และห้ามส่ง Supabase service-role key ให้ QA

### ทางเลือก C: Deploy บน staging host

เหมาะเมื่อจะทดสอบหลายคนหรือหลายวัน:

- deploy Docker image บน VM/staging
- ใช้ private network หรือ reverse proxy ที่มี TLS/auth
- เก็บ logs และ resource metrics บน host
- แยก `.env` และ database จาก production
- กำหนด rate limit และ upload size

ไม่ควรใช้เครื่อง developer เป็น public server สำหรับ long-running QA เพราะ performance และ availability จะปนกับงานอื่น

---

## 7. Stage A: Performance test procedure

### PERF-00 Smoke

- warm-up 3-5 requests
- ตรวจ HTTP status, JSON และ server logs
- ยังไม่นับ latency ชุดนี้เป็น baseline

### PERF-01 Baseline sequential

- concurrency `1`
- valid requests อย่างน้อย `30`
- ใช้ภาพ small, medium, large สลับกัน
- รายงาน average, p50, p95, p99, max และ throughput

### PERF-02 Size workload

- concurrency `1`
- อย่างน้อย 20 requests ต่อกลุ่ม: small, medium, large, poor_quality
- แยกผลตาม resolution และ file size

### PERF-03 Concurrency

- concurrency `2`, `4` และ `8`
- อย่างน้อย 50 requests ต่อระดับ
- หยุดที่ `4` หาก RAM/VRAM ไม่พอ
- บันทึก timeout, error, queueing และ latency degradation

### PERF-04 Error handling

ทดสอบ JPEG/PNG/WebP, content type ผิด, ไฟล์เกิน 10 MB, ไฟล์เสีย และ request ไม่มี image

คาดหวัง invalid request เป็น 4xx/413 ไม่ใช่ 500 และ server ต้องตอบ request ถัดไปได้

### Performance pass baseline

- valid-image HTTP 5xx เป็น 0
- valid-image timeout เป็น 0 ที่ concurrency 1 และ 2
- error rate valid workload ไม่เกิน 1%
- server ไม่ crash หลังจบ stress
- p50/p95/p99 และ throughput ต้องมีครบ
- memory ไม่เพิ่มต่อเนื่องแบบผิดปกติหลัง batch

ไม่กำหนด latency ตายตัวข้ามเครื่อง ให้ล็อก PERF-01 เป็น baseline ของ environment นั้น

---

## 8. Stage B: Functional validation เพื่อ calibrate

เริ่ม Stage B ได้เมื่อ Stage A ผ่านและล็อก build/environment แล้วเท่านั้น

### 8.1 Dataset ที่ต้องใช้

ใช้เอกสาร [QA_VALIDATION_TESTING_DATASET_REQUIREMENTS.md](QA_VALIDATION_TESTING_DATASET_REQUIREMENTS.md) เป็น requirement ของภาพจริงและ manifest

ต้องมี:

- reference image และ query image
- category ทั้ง 6 กลุ่ม
- exact/near duplicate
- different product และ same category
- resized, cropped, rotated, mirrored
- screenshot, watermark, AI artifact
- blur, dark, low contrast, low resolution
- Ground Truth decision และ label segmentation/detector

### 8.2 แบ่งข้อมูล

- Calibration set: 60%
- Validation set: 20%
- Final Testing set: 20%

Final Testing set ต้องแยกเก็บและห้ามนำผลให้ทีมพัฒนาระหว่างปรับค่า

### 8.3 วิธีรัน

1. reset reference store ก่อนแต่ละ batch
2. สร้าง reference จาก `reference_image`
3. ส่ง query ผ่าน endpoint เดิม
4. เก็บ response JSON เต็มฉบับ
5. เก็บ actual decision, similarity, quality score, detector flags, HTTP status และ latency
6. เทียบ actual กับ Ground Truth
7. สรุป confusion matrix, precision, recall, F1 และ false `DUPLICATE`

### 8.4 วิธี calibrate

1. วิเคราะห์ error จาก Calibration set แยกตาม category/scenario
2. ปรับทีละกลุ่ม: quality gate -> segmentation -> detector -> similarity threshold
3. รันซ้ำบน Calibration set
4. ตรวจผลบน Validation set
5. เลือกค่าที่ลด false `DUPLICATE` โดยไม่ทำให้ `REVIEW` สูงเกินไป
6. ล็อก config และ code version
7. รัน Final Testing set ครั้งเดียว
8. บันทึก before/after report และเหตุผลทุกค่าที่เปลี่ยน

ห้ามใช้ Final Testing set เพื่อเลือก threshold

### 8.5 Metrics สำคัญสำหรับ calibration

- precision/recall/F1 ต่อ `DUPLICATE`, `REVIEW`, `UNIQUE`
- false positive rate ของ `DUPLICATE`
- false negative rate ของ `DUPLICATE`
- review rate
- segmentation pass/fail rate
- detector precision/recall
- metrics แยกตาม category และ scenario
- latency p50/p95 เพื่อยืนยันว่า calibration ไม่ทำให้ performance แย่เกินไป

---

## 9. Report ที่ต้องส่งกลับ

```text
qa_result/
  stage_a_performance/
    requests.csv
    summary.xlsx
    environment.txt
    client.log
    server.log
  stage_b_validation/
    results.csv
    summary.xlsx
    raw_responses/
    confusion_matrix.csv
    calibration_notes.md
  README.txt
```

### Stage A requests.csv

```csv
run_id,scenario_id,image_id,concurrency,request_number,http_status,response_time_ms,response_size_bytes,error_type
RUN001,PERF-01,IMG_001,1,1,200,8421,1234,
RUN001,PERF-03,IMG_002,4,1,200,31240,1234,
```

### Stage B results.csv

```csv
case_id,category,scenario,ground_truth_decision,actual_decision,foreground_similarity,background_similarity,phash_similarity,detector_flags,latency_ms,qa_match,qa_comment
MOB_EXACT_001,MOBILE_TABLET,EXACT_DUPLICATE,DUPLICATE,DUPLICATE,0.991,0.963,1.0,NONE,8421,PASS,
```

ใน `calibration_notes.md` ต้องระบุ:

- ค่า config ก่อนและหลัง
- dataset split ที่ใช้เลือกค่า
- เหตุผลที่เปลี่ยน
- metric ก่อน/หลัง
- regression ที่พบ
- เหตุผลที่อนุมัติหรือไม่อนุมัติค่าใหม่

---

## 10. ความปลอดภัยและการล้างระบบหลังทดสอบ

- ห้ามเปิด public endpoint ถาวรจากเครื่อง developer
- ห้ามส่ง secret ใน `.env`, log หรือ report
- ใช้ภาพที่ได้รับอนุญาตให้ทดสอบ
- ลบ firewall rule, ปิด tunnel และหยุด container หลังเสร็จ
- ถ้าใช้ Supabase ให้แยก project/bucket สำหรับ test
- `InMemoryReferenceStore` จะหายเมื่อ process/container restart; ต้องบันทึก run order ให้ชัดเจน
