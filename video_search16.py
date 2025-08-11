import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from scipy.spatial import distance as dist

# === CONFIGURATION ===
REFERENCE_FOLDER = "headshots"
VIDEO_PATH = "FreedomKids1.mp4"
DISTANCE_THRESHOLD = 0.8   # Lower is stricter (tune this)
FRAME_JUMP = 50
MAX_FRAMES = 5000
OUTPUT_FOLDER = "matched_faces"
REF_OUTPUT = "reference_faces"

# === SETUP ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REF_OUTPUT, exist_ok=True)

# === INIT INSIGHTFACE ===
app = FaceAnalysis(name="buffalo_l")  # ArcFace-based model
app.prepare(ctx_id=0, det_size=(640, 640))

# === BLUR DETECTION ===
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold  # True if blurry

# === SECONDARY GEOMETRY CHECK ===
def facial_geometry_ok(landmarks):
    # landmarks: array [[x0, y0], [x1, y1], ...]
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    mouth_left = landmarks[3]
    mouth_right = landmarks[4]

    eye_dist = dist.euclidean(left_eye, right_eye)
    mouth_dist = dist.euclidean(mouth_left, mouth_right)

    # Simple sanity check: mouth-to-eye ratio within range
    ratio = mouth_dist / eye_dist if eye_dist > 0 else 0
    return 0.8 < ratio < 1.5

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
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cropped_face = frame[y1:y2, x1:x2]

        # === FILTER: blurry or incomplete faces ===
        #if is_blurry(cropped_face) or cropped_face.shape[0] < 50 or cropped_face.shape[1] < 50:
        #    print("   🚫 Face skipped (blurry or too small)")
        #    continue

        # === STEP 1: EMBEDDING SIMILARITY ===
        emb = face.normed_embedding
        distance_val = np.linalg.norm(emb - ref_embedding)
        print(f"   ➡️ Distance: {distance_val:.3f}")

        if distance_val < DISTANCE_THRESHOLD:
            # === STEP 2: SECONDARY GEOMETRY CHECK ===
            if facial_geometry_ok(face.landmark_2d_106):
                output_path = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count:04d}_match.jpg")
                cv2.imwrite(output_path, cropped_face)
                print(f"   ✅ Match saved: {output_path}")
            else:
                print("   ⚠️ Failed geometry verification")
        else:
            print("   ❌ Not a match")

cap.release()
print("✅ Done")
