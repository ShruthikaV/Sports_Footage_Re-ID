import sys
sys.path.append('./sort')
from sort import Sort

import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque

# Load model and class names
model = YOLO("models/best.pt")
CLASS_LABELS = model.model.names

cap = cv2.VideoCapture("15sec_input_720p.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output/reid_output.mp4", fourcc, fps, (width, height))

# Tracker
tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)

# Stable ID memory
next_stable_id = 1
stable_id_memory = {}  # stable_id: {'features': deque, 'last_seen': int, 'position': (x1, y1, x2, y2)}

# Parameters
HIST_BUFFER_SIZE = 5
DISAPPEAR_THRESHOLD_FRAMES = 60
FEATURE_THRESHOLD = 0.4
BALL_MOTION_THRESHOLD = 10  # pixels

# Ball tracking (only draw if ball moves)
last_ball_position = None

# Functions
def extract_color_histogram(img_crop):
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def average_hist_similarity(hist, hist_list):
    scores = [cv2.compareHist(hist, h, cv2.HISTCMP_BHATTACHARYYA) for h in hist_list]
    return sum(scores) / len(scores)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Detect all objects
    results = model(frame)
    detections = []
    players_boxes = []
    balls_boxes = []
    referees_boxes = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = CLASS_LABELS[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label.lower() == "player" and conf > 0.5:
                detections.append([x1, y1, x2, y2, conf])
                players_boxes.append((x1, y1, x2, y2))

            elif label.lower() == "referee" and conf > 0.5:
                referees_boxes.append((x1, y1, x2, y2))

            elif label.lower() == "ball" and conf > 0.5:
                balls_boxes.append((x1, y1, x2, y2))

    dets = np.array(detections) if detections else np.empty((0, 5))
    tracks = tracker.update(dets)

    # Player Re-ID
    assigned_ids = set()
    for track in tracks:
        x1, y1, x2, y2, _ = map(int, track)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        hist = extract_color_histogram(crop)

        matched_id = None
        best_score = FEATURE_THRESHOLD

        for sid, data in stable_id_memory.items():
            if sid in assigned_ids:
                continue
            score = average_hist_similarity(hist, data['features'])
            if score < best_score:
                matched_id = sid
                best_score = score

        if matched_id is None:
            matched_id = next_stable_id
            next_stable_id += 1
            stable_id_memory[matched_id] = {
                'features': deque(maxlen=HIST_BUFFER_SIZE),
                'last_seen': frame_count,
                'position': (x1, y1, x2, y2)
            }

        # Update memory
        stable_id_memory[matched_id]['features'].append(hist)
        stable_id_memory[matched_id]['last_seen'] = frame_count
        stable_id_memory[matched_id]['position'] = (x1, y1, x2, y2)
        assigned_ids.add(matched_id)

        # Draw Player box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
        cv2.putText(frame, f"Player {matched_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 255), 2)

    # Referee box
    for (x1, y1, x2, y2) in referees_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "Referee", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Ball tracking - only show if moving
    for (x1, y1, x2, y2) in balls_boxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if last_ball_position is not None:
            dist = np.linalg.norm(np.array([cx, cy]) - np.array(last_ball_position))
        else:
            dist = BALL_MOTION_THRESHOLD + 1  # ensure first frame shows ball

        if dist > BALL_MOTION_THRESHOLD:
            # Draw moving ball
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, "Ball", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            last_ball_position = (cx, cy)
        break  # only handle 1 ball per frame

    # Cleanup memory
    to_delete = []
    for sid, data in stable_id_memory.items():
        if frame_count - data['last_seen'] > DISAPPEAR_THRESHOLD_FRAMES:
            to_delete.append(sid)
    for sid in to_delete:
        del stable_id_memory[sid]

    out.write(frame)
    cv2.imshow("Re-ID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Done. Saved to output/reid_output.mp4 | Frames: {frame_count}")
 