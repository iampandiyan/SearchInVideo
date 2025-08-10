from deepface import DeepFace
import os
import json

dataset_dir = "dataset"  # subfolders per child
embeddings = []

for child_name in os.listdir(dataset_dir):
    child_path = os.path.join(dataset_dir, child_name)
    if os.path.isdir(child_path):
        for img_name in os.listdir(child_path):
            img_path = os.path.join(child_path, img_name)
            try:
                embedding = DeepFace.represent(img_path=img_path, model_name="Facenet",enforce_detection=False)[0]["embedding"]
                embeddings.append({"name": child_name, "embedding": embedding})
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

# Save embeddings
with open("kids_embeddings.json", "w") as f:
    json.dump(embeddings, f)
print("✅ Training complete. Embeddings saved.")
