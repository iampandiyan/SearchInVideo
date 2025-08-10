import cv2
from deepface import DeepFace
import numpy as np
import os

# === CONFIGURATION ===
REFERENCE_FOLDER = "headshots"
VIDEO_PATH = "FreedomKids1.mp4"
MODEL_NAME = "ArcFace"
DISTANCE_THRESHOLD = 6.0
MAX_FRAMES = 100
OUTPUT_FOLDER = "matched_frames"

# === SETUP OUTPUT FOLDER ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === LOAD AND AVERAGE EMBEDDINGS ===
def load_reference_embeddings(folder_path):
    embeddings = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            emb = DeepFace.represent(img, model_name=MODEL_NAME)[0]["embedding"]
            embeddings.append(emb)
    return np.mean(embeddings, axis=0)

ref_embedding = load_reference_embeddings(REFERENCE_FOLDER)

# === OPEN VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

while cap.isOpened() and frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 50
    print(f"\n📽️ Processing Frame {frame_count}")
    matched = False  # Flag to track if any match occurs

    # === DETECT FACES ===
    results = DeepFace.extract_faces(frame, enforce_detection=False)

    for face in results:
        face_img = face["face"]
        face_embedding = DeepFace.represent(face_img, model_name=MODEL_NAME)[0]["embedding"]

        # === CALCULATE DISTANCE ===
        distance = np.linalg.norm(np.array(face_embedding) - ref_embedding)
        print(f"   ➡️ Distance to reference: {distance:.2f}")

        # === MATCHING LOGIC ===
        if distance < DISTANCE_THRESHOLD:
            label = f"Match ({distance:.2f})"
            color = (0, 255, 0)
            matched = True
        else:
            label = f"No Match ({distance:.2f})"
            color = (0, 0, 255)

        # === DRAW BOX AND LABEL ===
        x, y, w, h = face["facial_area"].values()
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # === SAVE MATCHED FRAME ===
    if matched:
        output_path = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"   📸 Saved matched frame to {output_path}")

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()