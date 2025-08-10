import os
import cv2
import numpy as np
from deepface import DeepFace

# -------------------------------
# CONFIGURATION
# -------------------------------
REFERENCE_FOLDER = "headshots/"
VIDEO_PATH = "FreedomKids1.mp4"
OUTPUT_FOLDER = "matched_frames/"
MODEL_NAME = "Facenet"
DISTANCE_THRESHOLD = 5  # Loosened for better matching
DETECTOR_BACKEND = "retinaface"  # More robust than 'opencv'

# -------------------------------
# STEP 1: Load Reference Embeddings
# -------------------------------
print("🔍 Step 1: Loading reference images...")
reference_images = [os.path.join(REFERENCE_FOLDER, img) for img in os.listdir(REFERENCE_FOLDER)]
reference_embeddings = []

for img_path in reference_images:
    try:
        embedding = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
        reference_embeddings.append(embedding)
        print(f"✅ Loaded embedding for: {os.path.basename(img_path)}")
    except Exception as e:
        print(f"⚠️ Failed to process {img_path}: {e}")

if not reference_embeddings:
    print("❌ No valid reference embeddings found. Exiting.")
    exit()

# -------------------------------
# STEP 2: Process Video and Annotate Matches
# -------------------------------
print("🎥 Step 2: Processing video...")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
match_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 5
    print(f"🧪 Processing frame {frame_count}...")

    try:
        # Detect faces and get bounding boxes
        analysis = DeepFace.analyze(
            img_path=frame,
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND
        )

        print(f"🔍 Detected {len(analysis)} face(s) in frame {frame_count}")

        for face in analysis:
            x, y, w, h = face["region"]["x"], face["region"]["y"], face["region"]["w"], face["region"]["h"]
            cropped_face = frame[y:y+h, x:x+w]

            # Get embedding for cropped face
            try:
                face_embedding = DeepFace.represent(img_path=cropped_face, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
            except Exception as e:
                print(f"⚠️ Failed to get embedding at frame {frame_count}: {e}")
                continue

            # Compare with reference embeddings
            for ref_emb in reference_embeddings:
                distance = np.linalg.norm(np.array(face_embedding) - np.array(ref_emb))
                print(f"   ➡️ Distance to reference: {distance:.2f}")
                if distance < DISTANCE_THRESHOLD:
                    match_count += 1
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Match ({distance:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Save annotated frame
                    output_path = os.path.join(OUTPUT_FOLDER, f"matched_frame_{frame_count}.jpg")
                    cv2.imwrite(output_path, frame)
                    print(f"✅ Match found at frame {frame_count}, saved to {output_path}")
                    break  # Stop checking other references once matched
    except Exception as e:
        print(f"⚠️ Error at frame {frame_count}: {e}")

cap.release()
print(f"🎉 Done! Total frames processed: {frame_count}, Matches found: {match_count}")