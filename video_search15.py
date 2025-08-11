import cv2
from deepface import DeepFace
import numpy as np
import os

# === CONFIGURATION ===
TRUE_REFERENCE_FOLDER = "true_headshots"
FALSE_REFERENCE_FOLDER = "false_headshots"
VIDEO_PATH = "FreedomKids1.mp4"

# Change model to Facenet512
MODEL_NAME = "Facenet512"
DISTANCE_THRESHOLD = 0.9  # Recommended for Facenet512
FRAME_JUMP = 50
MAX_FRAMES = 5000

OUTPUT_FOLDER = "matched_faces"
TRUE_REF_OUTPUT = "reference_faces_true"
FALSE_REF_OUTPUT = "reference_faces_false"

# === SETUP OUTPUT FOLDERS ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TRUE_REF_OUTPUT, exist_ok=True)
os.makedirs(FALSE_REF_OUTPUT, exist_ok=True)

# === LOAD AND AVERAGE EMBEDDINGS ===
def load_reference_embeddings(folder_path, save_folder):
    embeddings = []
    ref_img_index = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            faces = DeepFace.extract_faces(img_path=img_path, enforce_detection=False)

            if not faces:
                print(f"⚠️ No face detected in {filename}, skipping.")
                continue

            face_img = faces[0]["face"]

            # Save cropped reference face
            ref_bgr = (face_img[:, :, ::-1] * 255).astype(np.uint8)
            ref_save_path = os.path.join(save_folder, f"ref_{ref_img_index:02d}.jpg")
            cv2.imwrite(ref_save_path, ref_bgr)
            print(f"💾 Saved reference face: {ref_save_path}")
            ref_img_index += 1

            emb = DeepFace.represent(face_img, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
            embeddings.append(emb)

    if not embeddings:
        raise ValueError(f"❌ No valid reference embeddings found in {folder_path}")

    return np.mean(embeddings, axis=0)

# === Load both TRUE and FALSE reference embeddings ===
true_ref_embedding = load_reference_embeddings(TRUE_REFERENCE_FOLDER, TRUE_REF_OUTPUT)
false_ref_embedding = load_reference_embeddings(FALSE_REFERENCE_FOLDER, FALSE_REF_OUTPUT)

# === HELPER: Normalize bounding box ===
def get_bbox(facial_area):
    if all(k in facial_area for k in ["x", "y", "w", "h"]):
        return facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
    elif all(k in facial_area for k in ["left", "top", "right", "bottom"]):
        x = facial_area["left"]
        y = facial_area["top"]
        w = facial_area["right"] - facial_area["left"]
        h = facial_area["bottom"] - facial_area["top"]
        return x, y, w, h
    else:
        raise ValueError(f"Unexpected facial_area format: {facial_area}")

# === OPEN VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

while cap.isOpened() and frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += FRAME_JUMP
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    print(f"\n📽️ Processing Frame {frame_count}")

    results = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
    for face in results:
        face_img = face["face"]
        facial_area = face["facial_area"]

        face_embedding = DeepFace.represent(face_img, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]

        # === CALCULATE DISTANCES ===
        distance_true = np.linalg.norm(np.array(face_embedding) - true_ref_embedding)
        distance_false = np.linalg.norm(np.array(face_embedding) - false_ref_embedding)

        print(f"   ✅ Distance to TRUE reference: {distance_true:.4f}")
        print(f"   ❌ Distance to FALSE reference: {distance_false:.4f}")

        # === MATCHING LOGIC ===
        if distance_true < DISTANCE_THRESHOLD and distance_true < distance_false:
            label = f"True Match ({distance_true:.4f})"
            color = (0, 255, 0)
        else:
            label = "False/No Match"
            color = (0, 0, 255)

        # === DRAW ON FRAME ===
        try:
            x, y, w, h = get_bbox(facial_area)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception as e:
            print(f"⚠️ Could not draw bbox: {e}")

        # === SAVE MATCHED FACE ===
        if "True Match" in label:
            output_path = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count:04d}_match.jpg")
            face_bgr = (face_img[:, :, ::-1] * 255).astype(np.uint8)
            cv2.imwrite(output_path, face_bgr)
            print(f"   📸 Saved matched face to {output_path}")

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
