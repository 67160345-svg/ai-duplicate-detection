# QA Localhost FastAPI Performance Test Requirements

เอกสารนี้กำหนดการทดสอบ **performance เท่านั้น** ของ prototype ที่รันบน localhost ผ่าน FastAPI

รอบนี้ไม่ใช้เพื่อสรุปความแม่นยำของ YOLO, OpenCLIP, pHash หรือ detector และไม่ต้องนำผลไป calibrate threshold ของโมเดล

## 1. เป้าหมาย

วัดความสามารถของ API เมื่อรับภาพจริงผ่าน endpoint:

```text
POST /gateway/detectDuplicateProduct
```

ต้องวัด:

- เวลา response ต่อ request
- throughput หรือจำนวน request ที่รองรับต่อวินาที
- ผลกระทบจาก concurrency
- error rate และ timeout
- memory/CPU usage
- พฤติกรรมเมื่อมีภาพขนาดและประเภทต่างกัน
- จุดที่ pipeline ใช้เวลามาก เช่น quality gate, detector, YOLO และ OpenCLIP

## 2. ขอบเขตระบบที่ทดสอบ

### รวมในรอบนี้

- FastAPI และ Uvicorn บนเครื่อง local
- multipart image upload
- quality gate
- screenshot/watermark/AI-artifact detector
- YOLOv8n-Seg
- OpenCLIP embedding
- pHash และ scoring
- `InMemoryReferenceStore`
- การตอบ HTTP status และ JSON response

### ไม่รวมในรอบนี้

- model accuracy, precision, recall, F1
- Ground Truth ว่า duplicate หรือ unique ถูกต้องหรือไม่
- การ calibrate threshold
- Supabase network latency และ pgvector performance
- ngrok, internet latency, gateway และ frontend
- production capacity หรือ SLA จริง
- `SPAM` behavior จาก seller/upload history

## 3. Test environment ที่ต้องบันทึก

QA ต้องบันทึกข้อมูลต่อไปนี้ใน `README.txt` ของผลทดสอบ:

- OS และ version
- CPU model และจำนวน logical cores
- RAM ทั้งหมดและ RAM ที่ว่างก่อนเริ่ม
- GPU model, VRAM และ CUDA version ถ้ามี
- Python version
- FastAPI/Uvicorn version
- YOLO/torch/OpenCLIP version
- commit หรือ build ที่ทดสอบ
- `REFERENCE_STORE` ต้องเป็น `memory`
- `YOLO_SEG_CONFIDENCE`
- `YOLO_SEG_IOU`
- `YOLO_SEG_MASK_CONFIDENCE`
- `YOLO_SEG_MAX_OBJECTS`
- คำสั่งที่ใช้ start server
- port ที่ใช้ทดสอบ

แนะนำให้รัน server แบบไม่ใช้ reload เพื่อไม่ให้ watcher มีผลต่อผลวัด:

```powershell
$env:REFERENCE_STORE="memory"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

ห้ามใช้ `--reload` ระหว่างเก็บผล performance อย่างเป็นทางการ

## 4. ชุดภาพสำหรับ workload

ภาพใช้เป็น input เพื่อสร้าง workload ไม่ใช่ dataset สำหรับตัดสินความถูกต้อง

โครงสร้างที่แนะนำ:

```text
performance_dataset/
  images/
    small/
    medium/
    large/
    poor_quality/
    detector_cases/
  manifest.csv
  README.txt
```

### 4.1 กลุ่มภาพขั้นต่ำ

| กลุ่ม | คุณสมบัติแนะนำ | จำนวนขั้นต่ำ |
|---|---|---:|
| `small` | 640x480 ถึง 1280x720 | 20 |
| `medium` | 1280x720 ถึง 1920x1080 | 20 |
| `large` | มากกว่า 1920x1080 หรือไฟล์ 5-10 MB | 10 |
| `poor_quality` | blur/dark/low contrast/low resolution | 10 |
| `detector_cases` | screenshot/watermark/AI-artifact หรือภาพปกติ | 10 |

รวมขั้นต่ำ 70 ภาพ และควรมีรูปแบบ JPEG, PNG และ WebP หากระบบรองรับ

ควรมีภาพจาก 6 กลุ่มสินค้าเดิมเพื่อให้ workload ใกล้เคียงการใช้งานจริง แต่ QA ไม่ต้อง label ว่าผลควรเป็น `DUPLICATE` หรือ `UNIQUE`

### 4.2 manifest.csv

ใช้ header ขั้นต่ำ:

```csv
image_id,image_path,category,workload_group,width,height,file_size_bytes,format
IMG_001,images/small/mobile_001.jpg,MOBILE_TABLET,small,1280,720,245678,jpeg
IMG_002,images/large/camera_001.jpg,CAMERA,large,4032,3024,7345678,jpeg
```

## 5. Preconditions

1. ใช้เครื่องเดียวกันระหว่างแต่ละ test run
2. ปิดงานหนักอื่น ๆ และไม่เปิด model training ระหว่างทดสอบ
3. warm up model ด้วย request อย่างน้อย 3-5 ครั้งก่อนเริ่มจับเวลา
4. ใช้ `REFERENCE_STORE=memory` และ reset process ก่อนเริ่มแต่ละ scenario
5. ไม่สรุปผลจาก request แรก เพราะรวมเวลา model loading
6. ตรวจว่า endpoint ตอบได้ด้วย HTTP 200/400/413 ตาม input ที่ตั้งใจทดสอบ
7. บันทึก log server แยกจาก client result
8. ใช้ไฟล์ภาพเดิมตลอดรอบ เพื่อให้เปรียบเทียบซ้ำได้

## 6. Test scenarios

### PERF-01 Baseline sequential

- concurrency: 1
- requests: 30
- ใช้ภาพ small, medium และ large สลับกัน
- เป้าหมาย: วัด baseline latency หลัง warm-up

### PERF-02 Sequential by image size

- concurrency: 1
- requests: อย่างน้อย 20 ต่อกลุ่มภาพ
- แยกรายงาน `small`, `medium`, `large`, `poor_quality`
- เป้าหมาย: ดูผลกระทบของ resolution และ file size

### PERF-03 Low concurrency

- concurrency: 2 และ 4
- requests: 50 ต่อระดับ
- เป้าหมาย: ดู throughput และ latency เมื่อมี request พร้อมกันเล็กน้อย

### PERF-04 Stress prototype

- concurrency: 8
- requests: 100
- timeout ต่อ request: 120 วินาที
- เป้าหมาย: หาจุดที่ latency เพิ่มสูงหรือเกิด error

ถ้าเครื่องมี RAM/VRAM จำกัด ให้หยุดที่ concurrency 4 และระบุเหตุผล ห้ามฝืนจนเครื่องล่ม

### PERF-05 Poor-quality fast path

- ใช้ภาพ `poor_quality` อย่างน้อย 20 requests
- เปรียบเทียบกับภาพปกติขนาดใกล้กัน
- เป้าหมาย: ยืนยันว่า quality gate ลดเวลา inference ได้หรือไม่

### PERF-06 Detector workload

- ใช้ภาพ `detector_cases` อย่างน้อย 20 requests
- แยก screenshot, watermark, AI-artifact และ normal image
- เป้าหมาย: วัด overhead ของ detector และ OCR ถ้ามี

### PERF-07 Upload/error handling

ทดสอบอย่างน้อย:

- ไฟล์ JPEG/PNG/WebP ที่ถูกต้อง
- content type ไม่ใช่ image
- ไฟล์เกิน 10 MB
- ไฟล์ภาพเสียหรือ decode ไม่ได้
- request ไม่มี image field

เป้าหมายคือวัด response time และตรวจว่า error ถูกคืนแบบ graceful ไม่ทำให้ server process หยุด

## 7. Metrics ที่ต้องเก็บ

ต่อ request ต้องเก็บ:

- `run_id`
- `scenario_id`
- `image_id`
- `concurrency`
- `request_number`
- `http_status`
- `response_time_ms`
- `response_size_bytes`
- `error_type`
- `server_decision` ถ้ามี

สรุปต่อ scenario:

- total requests
- successful requests
- failed requests
- timeout count
- error rate
- min latency
- average latency
- p50 latency
- p90 latency
- p95 latency
- p99 latency
- max latency
- throughput requests/second
- CPU average และ peak
- RAM average และ peak
- GPU utilization/VRAM ถ้ามี

หากทำได้ ให้บันทึก timestamp ของแต่ละ request และ server log เพื่อเทียบ client latency กับ processing time

## 8. เกณฑ์ผ่านรอบ Prototype

เกณฑ์นี้เป็น baseline สำหรับ localhost เท่านั้น ไม่ใช่ production SLA:

- baseline sequential ไม่มี crash
- HTTP 5xx ใน valid-image workload ต้องเป็น 0
- valid-image timeout ต้องเป็น 0 ใน concurrency 1 และ 2
- error rate ของ valid-image workload ไม่เกิน 1%
- server ต้องยังตอบ health/docs ได้หลัง stress test
- process ต้องไม่หยุดและ memory ต้องไม่เพิ่มต่อเนื่องผิดปกติหลังจบ batch
- report ต้องมี p50/p95/p99 latency และ throughput ครบทุก scenario
- invalid input ต้องได้ HTTP 4xx/413 ตามที่คาด ไม่ใช่ HTTP 500

ไม่กำหนดตัวเลข latency เดียวสำหรับทุกเครื่อง เพราะ YOLO/torch/OpenCLIP ขึ้นกับ CPU/GPU และ memory ของเครื่อง QA ให้ใช้ baseline ของ PERF-01 เป็นตัวเทียบระหว่างการปรับโค้ด

## 9. รูปแบบผลลัพธ์ที่ QA ต้องส่งกลับ

```text
qa_performance_result/
  requests.csv
  summary.xlsx
  environment.txt
  README.txt
  client.log
  server.log
```

`requests.csv` ต้องมีอย่างน้อย:

```csv
run_id,scenario_id,image_id,concurrency,request_number,http_status,response_time_ms,response_size_bytes,error_type
RUN001,PERF-01,IMG_001,1,1,200,8421,1234,
RUN001,PERF-01,IMG_002,1,2,200,7910,1234,
RUN002,PERF-07,bad_001,1,1,400,12,98,invalid_content_type
```

`summary.xlsx` ต้องมี sheet:

- `Environment`
- `Scenario_Summary`
- `Latency`
- `Throughput`
- `Resource_Usage`
- `Errors`

## 10. การนำผลไปใช้ต่อ

ผลรอบนี้ใช้เพื่อ:

1. เปรียบเทียบ performance ก่อนและหลังแก้ YOLO/detector
2. ระบุ bottleneck ว่าอยู่ที่ quality gate, detector, YOLO, OpenCLIP หรือ store
3. กำหนดค่า concurrency ที่เหมาะสมสำหรับ prototype
4. วางแผน timeout และ resource limit
5. ตัดสินใจว่าต้องทำ batch queue หรือ worker แยกหรือไม่

ผลรอบนี้ **ยังไม่ใช้** เพื่อ:

- เปลี่ยน `0.95` หรือ `0.70` threshold
- สรุปว่า model detect ถูกหรือผิด
- อ้างอิง production capacity
- สรุปว่า category ใดแม่นยำกว่า category อื่น
