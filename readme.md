# PPE Detection using YOLO

ระบบตรวจจับการใส่อุปกรณ์ป้องกันส่วนบุคคล (PPE) ด้วยโมเดล YOLO  
รองรับการตรวจครบต่อบุคคล เช่น Mask, Glove, Head Cover, PPE Coverall, Safety Shoes

## 📂 โครงสร้างโปรเจกต์

```
ppe_yolo/
│
├─ main.py                 # จุดเริ่มรันระบบ
├─ config.py               # ค่าการตั้งค่าและค่าคงที่
├─ detection.py            # โหลดโมเดล YOLO + รัน inference
├─ geometry.py             # ฟังก์ชันคำนวณพื้นที่, containment
├─ ppe_logic.py            # จัดการจับคู่ PPE ต่อบุคคล และตรวจครบ/ขาด
├─ drawing.py              # วาดกรอบและข้อความบนเฟรม
├─ camera.py               # จัดการกล้องและการอ่านเฟรม
├─ requirements.txt        # รายการ dependencies
│
├─ weights/
│   └─ PPE.pt               # โมเดลที่เทรนแล้ว
```

## 📦 การติดตั้ง

1. **โคลนโปรเจกต์**
```bash
git clone https://github.com/yourusername/ppe_yolo.git
cd ppe_yolo
```

2. **ติดตั้ง dependencies**
```bash
pip install -r requirements.txt
```

3. **วางไฟล์โมเดล**
   - วางไฟล์ `PPE.pt` ที่เทรนแล้วไว้ในโฟลเดอร์ `weights/`

## ▶️ การรันระบบ

```bash
python main.py
```

ระบบจะเปิดกล้องเว็บแคม (หรือ RTSP ถ้าปรับใน `main.py`) และแสดงการตรวจ PPE ต่อบุคคล  
กด `q` เพื่อออกจากโปรแกรม

## ⚙️ การตั้งค่า

ปรับค่าใน `config.py` ได้ เช่น:

- `REQUIRED_CLASSES` : เซ็ต PPE ที่ต้องมีครบ
- `PERSON_ALIASES` : ชื่อคลาสสำหรับคน (เช่น "human", "person")
- `CONF_THRESH` : ค่าความมั่นใจต่ำสุดที่รับ
- `MODEL_CONF` : ค่าความมั่นใจที่ส่งเข้า YOLO ตอนรัน
- `CONTAINMENT_RATIO` : สัดส่วน PPE ที่ต้องอยู่ในกรอบคน
- `PERSON_PAD_PX` : ระยะขยายกรอบคน

## 🖼️ การตรวจแบบเฟรมเว้นเฟรม

ใน `main.py` ปรับค่า:

```python
infer_every_n = 1   # ตรวจทุกเฟรม
infer_every_n = 2   # ตรวจทุก 2 เฟรม
```

ถ้าใช้ค่ามากกว่า 1 จะลดโหลดการประมวลผล แต่มีโอกาสพลาดวัตถุที่ผ่านเร็ว

## 📌 ฟีเจอร์

- ตรวจจับ PPE ต่อบุคคล
- ปรับกล้องเป็นเว็บแคมหรือ RTSP ได้
- เฟรมเว้นเฟรมเพื่อลดโหลด
- จัดโค้ดเป็นโมดูล ดูแลแก้ไขง่าย

---
