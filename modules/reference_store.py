"""Module: reference_store.py

จุดประสงค์: แยก "ที่เก็บภาพอ้างอิง" (storage/retrieval) ออกจาก business logic
ของ scoring_engine.py โดยสิ้นเชิง

ฝั่งอัลกอริทึม (scoring_engine.py) รับแค่ List[Dict] มาเทียบ ไม่สนใจว่าข้อมูลนั้น
มาจากไหน — ดังนั้นทีมที่เชื่อมต่อ DB จริงแค่ implement คลาสใหม่ตาม interface
`ReferenceImageStore` นี้ แล้วสลับ instance ที่ใช้ใน main.py จุดเดียว ไม่ต้องแตะ
scoring_engine.py หรือ feature_extractor.py เลย

สถานะปัจจุบัน:
- มีแค่ `InMemoryReferenceStore` สำหรับ dev/test เท่านั้น (พฤติกรรมเดิมเป๊ะๆ กับ
  DB_REFERENCE_IMAGES ที่เคยเป็น global list ตรงๆ ใน main.py) — ห้ามใช้ตัวนี้ใน
  production เพราะข้อมูลหายเมื่อ restart และไม่มี index จริง
- ทีม API ต้อง implement `PgVectorReferenceStore(ReferenceImageStore)` ที่คุยกับ
  Postgres+pgvector จริง ตาม schema ที่อยู่ใน schema.sql

หมายเหตุเรื่อง scope: พารามิเตอร์ `scope` ใน find_candidates ยังเป็น placeholder
เพราะยังไม่ได้ข้อสรุปจากทีม product ว่า "ซ้ำ" หมายถึงเทียบกับทั้งแพลตฟอร์ม, เฉพาะ
ร้านเดียวกัน, หรือเฉพาะหมวดหมู่เดียวกัน — เมื่อได้คำตอบแล้วแค่ส่ง dict เพิ่ม เช่น
{"seller_id": "..."} เข้ามา ไม่ต้องแก้ signature
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class ReferenceImageStore(ABC):
  """Interface ที่ business logic ทั้งหมดพึ่งพา (dependency ที่ main.py ต้อง inject)"""

  @abstractmethod
  def find_candidates(
      self,
      new_fg_vector: List[float],
      new_bg_vector: List[float],
      new_phash: str,
      top_k: int = 20,
      scope: Optional[Dict] = None,
  ) -> List[Dict]:
    """คืนค่า candidate records ที่ "น่าจะใกล้เคียงที่สุด" อย่างมาก top_k รายการ
    (ไม่ใช่ทุก record ในระบบ — เพื่อให้ scale ได้เมื่อข้อมูลเยอะ) แล้วส่งต่อให้
    scoring_engine.evaluate_baseline_decision() ตัดสินใจแบบละเอียดอีกที

    บน pgvector ของจริง ขั้นนี้ควรทำด้วย
        ORDER BY fg_vector <=> $1 LIMIT top_k
    ไม่ใช่ดึงทั้งตารางมาแล้วเทียบใน Python เหมือนตอนนี้

    scope: dict สำหรับ filter ขอบเขตการเทียบ เช่น {"seller_id": "..."} หรือ
           {"category": "..."} — ปัจจุบันยังไม่มีการ filter จริง (รอคำตอบ
           เรื่อง business scope) ถ้า scope=None คือเทียบกับทั้งหมดในระบบ

    แต่ละ record ที่คืนกลับมาต้องมี key อย่างน้อย:
        product_id, image_url, phash, fg_vector, bg_vector
    """
    raise NotImplementedError

  @abstractmethod
  def add(self, record: Dict) -> None:
    """บันทึก record ใหม่ลง store หลังประมวลผลภาพเสร็จ

    record ต้องมี key อย่างน้อย:
        product_id, image_url, phash, fg_vector, bg_vector
    """
    raise NotImplementedError


class InMemoryReferenceStore(ReferenceImageStore):
  """สำหรับ dev/test/POC เท่านั้น — เก็บใน RAM ตรงๆ ไม่มี persistence และ
  ไม่มี approximate nearest neighbor index จริง (คืนทุก record ที่ match scope
  ให้ scoring_engine ไป loop เทียบเองแบบ linear scan เหมือนพฤติกรรมเดิม)

  top_k ในคลาสนี้ "ไม่ได้ตัดจริง" เพราะไม่มีการจัดอันดับล่วงหน้า — พารามิเตอร์นี้
  มีไว้ให้ signature ตรงกับ interface เท่านั้น เมื่อสลับไปใช้ PgVectorReferenceStore
  ค่อยมีผลจริงผ่าน SQL LIMIT

  ห้ามใช้คลาสนี้ใน production
  """

  def __init__(self):
    self._records: List[Dict] = []

  def find_candidates(
      self,
      new_fg_vector: List[float],
      new_bg_vector: List[float],
      new_phash: str,
      top_k: int = 20,
      scope: Optional[Dict] = None,
  ) -> List[Dict]:
    if not scope:
      return list(self._records)
    return [
        r for r in self._records
        if all(r.get(key) == value for key, value in scope.items())
    ]

  def add(self, record: Dict) -> None:
    self._records.append(record)
