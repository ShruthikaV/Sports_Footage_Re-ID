# Re-Identification Report

## ✅ Objective
Detect players and keep their IDs consistent even if they leave and re-enter the frame.

## 🧠 Approach
- Used YOLOv11 (provided best.pt) for detection.
- Used SORT algorithm for tracking and re-identification.
- Drew bounding boxes and ID labels for visualization.

## ⚠️ Challenges
- Player overlap or occlusion can confuse the tracker.
- SORT doesn’t use appearance features.

## 🧪 Improvements
- Add DeepSORT to improve tracking based on player appearance.
- Fine-tune thresholds for better ID persistence.

## 📉 Challenges

- YOLO sometimes detected multiple referees or balls.
- Histogram matching is sensitive to lighting changes.
- Motion-based ball filtering needed fine-tuning to eliminate false positives.

## 👨‍💻 Tools Used

- Python, OpenCV, NumPy
- Ultralytics YOLOv11
- SORT tracker
- Histogram-based matching

## 📦 Files Submitted

- `main.py`: Main code
- `models/best.pt`: Detection weights (provided)
- `sort/`: Tracking module
- `output/reid_output.mp4`: Final video output
- `README.md`, `report.md`, `requirements.txt`

## ✨ Final Note

A consistent and modular pipeline was built for re-ID and selective detection. All code is reproducible, easy to test, and designed with real-world deployment in mind.