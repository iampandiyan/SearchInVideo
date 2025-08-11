import cv2
from deepface import DeepFace
import os

# --- CONFIG ---
video_path = "FreedomKids1.mp4"       # your video file
output_dir = "faces_output"    # where cropped faces will be saved
frame_skip = 5                 # process every nth frame to speed up

# Create output folder if not exists
os.makedirs(output_dir, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    try:
        detections = DeepFace.extract_faces(frame, detector_backend="retinaface", enforce_detection=False)
        
        for idx, face in enumerate(detections):
            facial_area = face["facial_area"]  # dict with x,y,w,h
            x = facial_area["x"]
            y = facial_area["y"]
            w = facial_area["w"]
            h = facial_area["h"]

            # Crop face
            cropped_face = frame[y:y+h, x:x+w]

            # Save image
            save_path = os.path.join(output_dir, f"face_{frame_count}_{idx}.jpg")
            cv2.imwrite(save_path, cropped_face)
            saved_count += 1

    except Exception as e:
        print(f"Error at frame {frame_count}: {e}")

cap.release()
print(f"✅ Done! {saved_count} faces saved in '{output_dir}'")
