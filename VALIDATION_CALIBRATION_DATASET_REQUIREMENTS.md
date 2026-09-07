# Functional Validation และ Calibration Dataset Requirements

เอกสารนี้ใช้เฉพาะ **Stage B หลัง Performance test ผ่านแล้ว** เพื่อวัดผล model pipeline และนำ report ไป calibrate ต่อ

## เป้าหมาย

วัดและปรับ:

- YOLOv8n-Seg foreground/background
- OpenCLIP foreground/background similarity
- pHash similarity/distance
- quality gate
- screenshot/watermark/AI-artifact detector
- decision `DUPLICATE`, `REVIEW`, `UNIQUE`, `INVALID_DATA`

## โครงสร้าง dataset

```text
validation_dataset/
  images/
    reference/
    query/
  manifest.csv
  README.txt
```

## หมวดสินค้า

ต้องครอบคลุม 6 กลุ่ม:

- `MOBILE_TABLET`
- `APPAREL_FASHION`
- `CAMERA`
- `TOOLS`
- `BEAUTY_HEALTH`
- `COMPUTER_IT`

หากไม่ใช่กลุ่มเป้าหมายให้ใช้ `UNKNOWN` และ expected decision เป็น `REVIEW`

## Scenario ขั้นต่ำ

- `EXACT_DUPLICATE`
- `NEAR_DUPLICATE`
- `RESIZED`
- `RECOMPRESSED`
- `CROPPED`
- `ROTATED`
- `MIRRORED`
- `DIFFERENT_PRODUCT`
- `SAME_CATEGORY_DIFFERENT_PRODUCT`
- `SIMILAR_BACKGROUND_DIFFERENT_PRODUCT`
- `MULTI_OBJECT`
- `UNSUPPORTED_OBJECT`
- `SCREENSHOT`
- `WATERMARK`
- `AI_IMAGE`
- `BLUR`
- `DARK`
- `LOW_CONTRAST`
- `LOW_RESOLUTION`

รอบแรกแนะนำอย่างน้อย 360 cases: 6 categories x 60 cases และมี reference product อย่างน้อย 5 รายการต่อ category

## manifest.csv

```csv
case_id,reference_image,query_image,category,scenario,ground_truth_decision,same_physical_product,expected_segmentation,expected_detector,quality_label,notes
MOB_EXACT_001,images/reference/mobile_001.jpg,images/query/mobile_001_exact.jpg,MOBILE_TABLET,EXACT_DUPLICATE,DUPLICATE,TRUE,PASS,NONE,GOOD,Same product
MOB_DIFF_001,images/reference/mobile_001.jpg,images/query/mobile_099.jpg,MOBILE_TABLET,DIFFERENT_PRODUCT,UNIQUE,FALSE,PASS,NONE,GOOD,Different product
APP_WM_001,images/reference/apparel_001.jpg,images/query/apparel_001_wm.jpg,APPAREL_FASHION,WATERMARK,REVIEW,TRUE,PASS,WATERMARK,GOOD,Visible text overlay
CAM_BLUR_001,,images/query/camera_blur.jpg,CAMERA,BLUR,REVIEW,FALSE,REVIEW,QUALITY,BLUR,Quality gate case
```

Required values:

- `ground_truth_decision`: `DUPLICATE`, `REVIEW`, `UNIQUE`, `INVALID_DATA`
- `expected_segmentation`: `PASS`, `FAIL`, `REVIEW`
- `expected_detector`: `NONE`, `SCREENSHOT`, `WATERMARK`, `AI_ARTIFACT`, `QUALITY`
- `quality_label`: `GOOD`, `BLUR`, `DARK`, `LOW_CONTRAST`, `LOW_RESOLUTION`, `OTHER`

## วิธีแบ่งข้อมูล

- Calibration set: 60%
- Validation set: 20%
- Final Testing set: 20%

Final Testing set ต้องเก็บแยกและห้ามส่งผลให้ทีมพัฒนาก่อนล็อกค่า

## วิธีรัน Stage B

1. ใช้ build/environment เดียวกับ Stage A
2. reset `REFERENCE_STORE=memory` ก่อนแต่ละ batch
3. สร้าง reference จาก `reference_image`
4. ส่ง `query_image` เข้า `/gateway/detectDuplicateProduct`
5. เก็บ response JSON เต็มฉบับ
6. บันทึก actual decision, similarity, quality scores, detector flags, HTTP status และ latency
7. เทียบ actual กับ Ground Truth
8. สรุปผลแยกตาม category และ scenario

## Metrics

- confusion matrix
- precision, recall, F1 ต่อ decision
- false positive rate ของ `DUPLICATE`
- false negative rate ของ `DUPLICATE`
- review rate
- segmentation pass/fail rate
- detector precision/recall
- latency p50/p95

เป้าหมาย prototype เริ่มต้น:

- ไม่มี false `DUPLICATE` ใน `DIFFERENT_PRODUCT` ที่ชัดเจน
- `DUPLICATE` precision อย่างน้อย 95%
- ภาพ quality เสียถูกส่ง `REVIEW` อย่างน้อย 90%

## ผลลัพธ์ที่ต้องส่ง

```text
stage_b_validation/
  results.csv
  summary.xlsx
  raw_responses/
    <case_id>.json
  confusion_matrix.csv
  calibration_notes.md
```

`results.csv` ต้องมี:

```csv
case_id,category,scenario,ground_truth_decision,actual_decision,foreground_similarity,background_similarity,phash_similarity,detector_flags,latency_ms,qa_match,qa_comment
MOB_EXACT_001,MOBILE_TABLET,EXACT_DUPLICATE,DUPLICATE,DUPLICATE,0.991,0.963,1.0,NONE,8421,PASS,
```

`calibration_notes.md` ต้องระบุค่า config ก่อน/หลัง, split ที่ใช้, เหตุผลที่เปลี่ยน, metrics ก่อน/หลัง และ regression ที่พบ

## Calibration workflow

1. วิเคราะห์ error บน Calibration set
2. ปรับทีละกลุ่ม: quality gate -> segmentation -> detector -> similarity threshold
3. รันซ้ำบน Calibration set
4. ตรวจซ้ำบน Validation set
5. ล็อก code และ config
6. รัน Final Testing set ครั้งเดียว
7. เก็บ before/after report

ห้ามเลือก threshold จาก Final Testing set
