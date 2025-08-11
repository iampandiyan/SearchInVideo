import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

#Good Accuracy Modeland working
# === CONFIGURATION ===
REFERENCE_FOLDER = "headshots"
VIDEO_PATH = "Sam1.mp4"
DISTANCE_THRESHOLD = 0.8   # Lower is stricter (tune this)
FRAME_JUMP = 10
MAX_FRAMES = 5000
OUTPUT_FOLDER = "matched_faces"
REF_OUTPUT = "reference_faces"
SKIPPED_FOLDER = "skipped_faces"
NOT_MATCHED_FOLDER = "not_matched_faces"

# === SETUP ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REF_OUTPUT, exist_ok=True)
os.makedirs(SKIPPED_FOLDER, exist_ok=True)
os.makedirs(NOT_MATCHED_FOLDER, exist_ok=True)

# === INIT INSIGHTFACE ===
app = FaceAnalysis(name="buffalo_l")  # ArcFace-based model
app.prepare(ctx_id=0, det_size=(640, 640))

# === BLUR DETECTION ===
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold  # True if blurry

# === LOAD REFERENCE EMBEDDINGS ===
def load_reference_embeddings(folder_path):
    embeddings = []
    idx = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            faces = app.get(img)

            if not faces:
                print(f"⚠️ No face in {filename}, skipping.")
                continue

            face = faces[0]
            emb = face.normed_embedding
            embeddings.append(emb)

            # Save cropped reference face
            x1, y1, x2, y2 = map(int, face.bbox)
            cropped = img[y1:y2, x1:x2]
            ref_save_path = os.path.join(REF_OUTPUT, f"ref_{idx:02d}.jpg")
            cv2.imwrite(ref_save_path, cropped)
            print(f"💾 Saved reference face: {ref_save_path}")
            idx += 1

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
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    print(f"\n📽️ Processing Frame {frame_count}")

    faces = app.get(frame)
    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = map(int, face.bbox)
        cropped_face = frame[y1:y2, x1:x2]

        # === FILTER: blurry or incomplete faces ===
        if is_blurry(cropped_face) or cropped_face.shape[0] < 50 or cropped_face.shape[1] < 50:
            print("   🚫 Face skipped (blurry or too small)")
            skip_path = os.path.join(SKIPPED_FOLDER, f"frame_{frame_count:04d}_face{idx}.jpg")
            cv2.imwrite(skip_path, cropped_face)
            continue

        # === EMBEDDING SIMILARITY ONLY ===
        emb = face.normed_embedding
        distance_val = np.linalg.norm(emb - ref_embedding)
        print(f"   ➡️ Distance: {distance_val:.3f}")

        if distance_val < DISTANCE_THRESHOLD:
            output_path = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count:04d}_face{idx}_match.jpg")
            cv2.imwrite(output_path, cropped_face)
            print(f"   ✅ Match saved: {output_path}")
        else:
            print("   ❌ Not a match")
            not_match_path = os.path.join(NOT_MATCHED_FOLDER, f"frame_{frame_count:04d}_face{idx}.jpg")
            cv2.imwrite(not_match_path, cropped_face)

cap.release()
print("✅ Done")
