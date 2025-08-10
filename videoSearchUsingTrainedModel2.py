import cv2
from deepface import DeepFace
import json
import numpy as np
import os

# Load trained embeddings (if needed later)
with open("kids_embeddings.json", "r") as f:
    stored_embeddings = json.load(f)

video_path = "FreedomKids1.mp4"
cap = cv2.VideoCapture(video_path)
frame_skip = 5
frame_count = 0

save_dir = "matched_screenshots"
os.makedirs(save_dir, exist_ok=True)

def brightness_ok(frame, threshold=50):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    if not brightness_ok(frame):
        continue  # skip dark frames

    try:
        detections = DeepFace.detectFace(
            frame, detector_backend="retinaface", enforce_detection=False
        )

        if detections is None or (isinstance(detections, np.ndarray) and detections.size == 0):
            continue

        # Search in dataset
        results_list = DeepFace.find(
            frame, db_path="kids_dataset", model_name="Facenet", enforce_detection=False
        )

        if isinstance(results_list, list) and len(results_list) > 0 and not results_list[0].empty:
            matched_identity = os.path.basename(results_list[0].iloc[0]['identity'])
            print(f"✅ Match found in frame {frame_count}: {matched_identity}")

            # Save screenshot
            save_path = os.path.join(
                save_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{frame_count}_{matched_identity}.jpg"
            )
            cv2.imwrite(save_path, frame)
            print(f"📸 Screenshot saved: {save_path}")

    except Exception as e:
        print("Detection error:", e)

cap.release()
print("✅ Video search complete.")
