import cv2
from deepface import DeepFace
import numpy as np
import os

# === CONFIGURATION ===
REFERENCE_FOLDER = "headshots"
VIDEO_PATH = "FreedomKids1.mp4"
MODEL_NAME = "ArcFace"
DISTANCE_THRESHOLD = 6.0  # Lower threshold for stricter matching
FRAME_JUMP = 50           # Skip frames to speed up
MAX_FRAMES = 5000         # Max frames to process
OUTPUT_FOLDER = "matched_faces"
REF_OUTPUT = "reference_faces"  # Folder to store cropped reference face(s)

# === SETUP OUTPUT FOLDERS ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REF_OUTPUT, exist_ok=True)

# === LOAD AND AVERAGE EMBEDDINGS (with cropping + saving reference face) ===
def load_reference_embeddings(folder_path):
    embeddings = []
    ref_img_index = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, filename)

            # Crop face from reference image
            faces = DeepFace.extract_faces(img_path=img_path, enforce_detection=False)
            if not faces:
                print(f"⚠️ No face detected in {filename}, skipping.")
                continue

            face_img = faces[0]["face"]  # Take the first detected face

            # Save cropped reference face image
            ref_bgr = (face_img[:, :, ::-1] * 255).astype(np.uint8)
            ref_save_path = os.path.join(REF_OUTPUT, f"ref_{ref_img_index:02d}.jpg")
            cv2.imwrite(ref_save_path, ref_bgr)
            print(f"💾 Saved reference face: {ref_save_path}")
            ref_img_index += 1

            # Create embedding
            emb = DeepFace.represent(face_img, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
            embeddings.append(emb)

    if not embeddings:
        raise ValueError("❌ No valid reference embeddings found.")

    return np.mean(embeddings, axis=0)

ref_embedding = load_reference_embeddings(REFERENCE_FOLDER)

# === OPEN VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

while cap.isOpened() and frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += FRAME_JUMP
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)  # Jump frames efficiently
    print(f"\n📽️ Processing Frame {frame_count}")

    # === DETECT FACES IN VIDEO FRAME ===
    results = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
    for face in results:
        face_img = face["face"]
        face_embedding = DeepFace.represent(face_img, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]

        # === CALCULATE DISTANCE ===
        distance = np.linalg.norm(np.array(face_embedding) - ref_embedding)
        print(f"   ➡️ Distance to reference: {distance:.2f}")

        # === MATCHING LOGIC ===
        if distance < DISTANCE_THRESHOLD:
            # Save cropped matched face
            output_path = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count:04d}_match.jpg")
            face_bgr = (face_img[:, :, ::-1] * 255).astype(np.uint8)
            cv2.imwrite(output_path, face_bgr)
            print(f"   📸 Saved matched face to {output_path}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
