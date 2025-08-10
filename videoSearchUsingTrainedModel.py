import cv2
import numpy as np
import pickle
import os
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
with open("child_embeddings.pkl", "rb") as f:
    embedding_db = pickle.load(f)

# Create results folder
os.makedirs("results", exist_ok=True)

# Video file
video_path = "FreedomKids1.mp4"
cap = cv2.VideoCapture(video_path)

frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    if frame_no % 30 == 0:  # process every ~1 second for 30fps video
        try:
            detected_faces = DeepFace.represent(
                img_path=frame,
                model_name="Facenet",
                enforce_detection=False
            )
            for face in detected_faces:
                face_embedding = np.array(face["embedding"]).reshape(1, -1)
                for child_name, embeddings in embedding_db.items():
                    for db_emb in embeddings:
                        sim = cosine_similarity(face_embedding, np.array(db_emb).reshape(1, -1))[0][0]
                        if sim > 0.7:  # adjust threshold for better accuracy
                            print(f"✅ Match found for {child_name} in frame {frame_no}, similarity: {sim:.2f}")
                            # Save screenshot
                            save_path = f"results/{child_name}_frame{frame_no}.jpg"
                            cv2.imwrite(save_path, frame)
                            print(f"📸 Screenshot saved: {save_path}")
                            break  # Stop checking other embeddings for this frame
        except Exception as e:
            print(f"Error: {e}")

cap.release()
print("🔍 Search completed.")
