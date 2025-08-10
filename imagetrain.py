from deepface import DeepFace
import os
import numpy as np
import pickle

# Path to your child images dataset
dataset_path = "dataset/"  # e.g., dataset/child_name/image1.jpg

embedding_db = {}

for child_name in os.listdir(dataset_path):
    embeddings = []
    child_folder = os.path.join(dataset_path, child_name)
    for img_file in os.listdir(child_folder):
        img_path = os.path.join(child_folder, img_file)
        embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        embeddings.append(embedding)
    embedding_db[child_name] = embeddings

# Save embeddings to file for Phase 2
with open("child_embeddings.pkl", "wb") as f:
    pickle.dump(embedding_db, f)

print("Embeddings saved!")
