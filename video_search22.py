import cv2
import numpy as np
import os
import shutil
import threading
from tkinter import Tk, Label, Entry, Button, filedialog, StringVar, DoubleVar
from tkinter import ttk
from insightface.app import FaceAnalysis


# === INIT INSIGHTFACE ===
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))


# === BLUR DETECTION ===
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold


# === LOAD REFERENCE EMBEDDINGS ===
def load_reference_embeddings(folder_path):
    embeddings = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            faces = app.get(img)
            if not faces:
                continue
            embeddings.append(faces[0].normed_embedding)
    if not embeddings:
        raise ValueError("❌ No valid reference embeddings found.")
    return np.mean(embeddings, axis=0)


# === PASS 1: Extract faces for manual review ===
def extract_faces():
    vid_folder = video_folder.get()
    output_faces_dir = os.path.join(output_folder.get(), "faces_extracted")
    skipped_folder = os.path.join(output_folder.get(), "faces_skipped")

    frame_jump_val = int(frame_jump.get())
    max_frames_val = int(max_frames.get())

    os.makedirs(output_faces_dir, exist_ok=True)
    os.makedirs(skipped_folder, exist_ok=True)

    video_files = [f for f in os.listdir(vid_folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    total_videos = len(video_files)
    progress_var.set(0)

    for i, video_name in enumerate(video_files, start=1):
        video_path = os.path.join(vid_folder, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened() and frame_count < max_frames_val:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += frame_jump_val
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            faces = app.get(frame)

            for idx, face in enumerate(faces):
                x1, y1, x2, y2 = map(int, face.bbox)
                cropped_face = frame[y1:y2, x1:x2]
                if cropped_face.size == 0:
                    continue
                if cropped_face.shape[0] < 50 or cropped_face.shape[1] < 50 or is_blurry(cropped_face):
                    cv2.imwrite(os.path.join(skipped_folder, f"{video_name}_frame{frame_count}_face{idx}.jpg"), cropped_face)
                else:
                    cv2.imwrite(os.path.join(output_faces_dir, f"{video_name}_frame{frame_count}_face{idx}.jpg"), cropped_face)

        cap.release()
        progress_var.set((i / total_videos) * 100)
        root.update_idletasks()

    print(f"[INFO] Faces saved for manual review in {output_faces_dir}")
    print(f"[ACTION] Copy correct faces into faces_reference/ before running Pass 2")


# === PASS 2: Match only validated reference faces ===
def match_faces():
    ref_folder = os.path.join(output_folder.get(), "faces_reference")
    matched_faces_folder = os.path.join(output_folder.get(), "matched_faces")
    not_matched_folder = os.path.join(output_folder.get(), "not_matched_faces")
    matched_videos_dir = matched_videos_folder.get()

    threshold = float(distance_threshold.get())
    frame_jump_val = int(frame_jump.get())
    max_frames_val = int(max_frames.get())

    for folder in [matched_faces_folder, not_matched_folder, matched_videos_dir]:
        os.makedirs(folder, exist_ok=True)

    ref_embedding = load_reference_embeddings(ref_folder)

    vid_folder = video_folder.get()
    video_files = [f for f in os.listdir(vid_folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    total_videos = len(video_files)
    progress_var.set(0)

    for i, video_name in enumerate(video_files, start=1):
        video_path = os.path.join(vid_folder, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        match_found = False

        while cap.isOpened() and frame_count < max_frames_val:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += frame_jump_val
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            faces = app.get(frame)

            for idx, face in enumerate(faces):
                x1, y1, x2, y2 = map(int, face.bbox)
                cropped_face = frame[y1:y2, x1:x2]
                if cropped_face.size == 0:
                    continue
                emb = face.normed_embedding
                dist = np.linalg.norm(emb - ref_embedding)
                if dist < threshold:
                    cv2.imwrite(os.path.join(matched_faces_folder, f"{video_name}_frame{frame_count}_face{idx}.jpg"), cropped_face)
                    match_found = True
                else:
                    cv2.imwrite(os.path.join(not_matched_folder, f"{video_name}_frame{frame_count}_face{idx}.jpg"), cropped_face)

            if match_found:
                shutil.move(video_path, os.path.join(matched_videos_dir, video_name))
                break

        cap.release()
        progress_var.set((i / total_videos) * 100)
        root.update_idletasks()

    print("[INFO] Matching complete.")


# === GUI ===
root = Tk()
root.title("Face Extraction & Matching Tool")
root.geometry("700x500")

reference_folder = StringVar()
video_folder = StringVar()
matched_videos_folder = StringVar()
output_folder = StringVar()
distance_threshold = StringVar(value="0.8")
frame_jump = StringVar(value="10")
max_frames = StringVar(value="5000")
progress_var = DoubleVar()

Label(root, text="Videos Folder:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
Entry(root, textvariable=video_folder, width=50).grid(row=0, column=1, padx=5, pady=5)
Button(root, text="Browse", command=lambda: video_folder.set(filedialog.askdirectory())).grid(row=0, column=2, padx=5, pady=5)

Label(root, text="Matched Videos Output Folder:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
Entry(root, textvariable=matched_videos_folder, width=50).grid(row=1, column=1, padx=5, pady=5)
Button(root, text="Browse", command=lambda: matched_videos_folder.set(filedialog.askdirectory())).grid(row=1, column=2, padx=5, pady=5)

Label(root, text="Output Base Folder:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
Entry(root, textvariable=output_folder, width=50).grid(row=2, column=1, padx=5, pady=5)
Button(root, text="Browse", command=lambda: output_folder.set(filedialog.askdirectory())).grid(row=2, column=2, padx=5, pady=5)

Label(root, text="Distance Threshold:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
Entry(root, textvariable=distance_threshold).grid(row=3, column=1, padx=5, pady=5)

Label(root, text="Frame Jump:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
Entry(root, textvariable=frame_jump).grid(row=4, column=1, padx=5, pady=5)

Label(root, text="Max Frames:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
Entry(root, textvariable=max_frames).grid(row=5, column=1, padx=5, pady=5)

Button(root, text="Pass 1: Extract Faces", command=lambda: threading.Thread(target=extract_faces).start()).grid(row=6, column=0, columnspan=3, pady=10)
Button(root, text="Pass 2: Match Faces", command=lambda: threading.Thread(target=match_faces).start()).grid(row=7, column=0, columnspan=3, pady=10)

progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.grid(row=8, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

for i in range(3):
    root.grid_columnconfigure(i, weight=1)

root.mainloop()
