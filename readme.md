# PPE Detection using YOLO

A system for detecting the use of Personal Protective Equipment (PPE) with the YOLO model.
Supports per-person PPE verification, such as Mask, Glove, Head Cover, PPE Coverall, and Safety Shoes.

## 📂 Project Structure

```
PPE/
│src/
│ ├─ main.py                 # The entry point of the system
│ ├─ config.py               # Configuration and constants
│ ├─ detection.py            # Load YOLO model + run inference
│ ├─ geometry.py             # Functions for area calculation, containment
│ ├─ ppe_logic.py            # Manage PPE matching to persons and check completeness/absence
│ ├─ drawing.py              # Draw bounding boxes and text on frames
│ ├─ camera.py               # Manage camera and frame reading
│ ├─ requirements.txt        # List of dependencies
│
│weights/
│ ├─ PPE.pt               # Trained model
│
defaults.yaml
```

## 📦 Installation

1. **Clone the project**
```bash
git clone https://github.com/yourusername/ppe_yolo.git
cd ppe_yolo
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Place the model file**
   - Place the trained model file `PPE.pt` in the `weights/` folder

## ▶️ Run the system

```bash
python main.py
```

The system will open the webcam (or RTSP if configured in `main.py`) and display PPE detection per person.
Press `q` to exit the program.

## ⚙️ Configuration

Adjust the values in `config.py` as needed:

- `REQUIRED_CLASSES` : Set PPE that must be complete
- `PERSON_ALIASES` : Class names for people (e.g., "human", "person")
- `CONF_THRESH` : Minimum confidence threshold
- `MODEL_CONF` : Confidence threshold for YOLO inference
- `CONTAINMENT_RATIO` : PPE ratio that must be inside the person's frame
- `PERSON_PAD_PX` : Frame padding for people

## 🖼️ Frame Skipping

Adjust the values in `main.py`:

```python
infer_every_n = 1   # Infer every frame
infer_every_n = 2   # Infer every 2 frames
```

If using a value greater than 1 will reduce processing load but may miss objects passing by quickly.

## 📌 Features

- Detect PPE per person
- Adjust camera to webcam or RTSP
- Frame skipping to reduce load
- Organize code into modules for easy maintenance

---
