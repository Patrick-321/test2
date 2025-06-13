# ✨ YOLO Magic Mirror Detection

A fun, interactive object detection system that uses YOLOv5 to detect people, cell phones, and cups — and overlays magical visual effects like sparkles, emojis, and animated borders in real-time using your webcam.

---

## 🎯 Features
### Targeted class: person, cell phone, and cup

- 🧙 Real-time YOLOv5 object detection (person, cell phone, cup)
- ✨ Animated rainbow borders and sparkle particles
- 💫 Floating emoji effects based on object class
- 🔮 Magical text overlays when objects are detected
- 📹 Automatically saves a 5-second video clip when target objects appear
- 🪄 Inspired by augmented reality "magic mirror" concepts

---

## 🛠 Setup Instructions

### 1. Clone this repo

```bash
git clone https://github.com/yourusername/yolov5-magic-mirror
cd yolov5-magic-mirror
```

### 2. Install Dependencies

Ensure Python 3.8+ is installed. Then:

```bash
pip install torch torchvision opencv-python pandas
```

> **Note**: YOLOv5 is auto-loaded via `torch.hub` (internet required on first run).

---

## ▶️ Run the Program

```bash
python WebCamSave_Yolo.py
```

Then:
- Show a **person**, **cell phone**, or **cup** to the webcam.
- Watch the magic effects appear!
- A 5-second video (`magical_yolo_detection.mp4`) will be saved when an object is detected.


### Not Recommended running this version:
not megical
```bash
python WebCamSave_Yolo_v3.py
```

---

## 📦 Model Details

- Model: `yolov5s` (small version for speed)
- Source: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- Target Classes: `person`, `cell phone`, `cup`

---



## ✨ Fun Effects Summary

- `add_rainbow_border`: Adds animated rainbow borders around detected objects.
- `add_sparkle_effect`: Generates particles that fade away with gravity.
- `add_floating_emojis`: Shows themed emoji above the object.
- `add_magic_text_effects`: Displays rotating magical messages on detection.

---

## 📂 Output

- Video output: `magical_yolo_detection.mp4`
- Duration: 5 seconds per detection
- Format: MP4, 20 FPS

---

## ❓ Tips

- Make sure your webcam is accessible.
- Avoid low-light environments for best detection accuracy.
- Press `Q` to exit the app anytime.

---

## 📄 License

MIT License © 2025 Zwei Zhou  
Powered by [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)

---

## ❤️ Acknowledgements

Thanks to:
- [Ultralytics](https://github.com/ultralytics/yolov5) for YOLOv5
- OpenCV for real-time computer vision tools
