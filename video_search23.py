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

# === LOAD REFERENCE EMBEDDINGS (with manual validation support) ===
def load_reference_embeddings(folder_path, ref_output):
    """
    Loads reference embeddings from:
    1. The main reference folder selected in the GUI
    2. An optional 'manually_selected_faces' subfolder inside it
    Saves cropped faces into ref_output.
    """
    embeddings = []
    idx = 0
    os.makedirs(ref_output, exist_ok=True)

    # Merge main ref folder + manual validation folder
    all_sources = [folder_path]
    manual_folder = os.path.join(folder_path, "manually_selected_faces")
    if os.path.exists(manual_folder):
        all_sources.append(manual_folder)

    for source in all_sources:
        if not os.path.exists(source):
            continue
        for filename in os.listdir(source):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(source, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue
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
                ref_save_path = os.path.join(ref_output, f"ref_{idx:02d}.jpg")
                cv2.imwrite(ref_save_path, cropped)
                print(f"💾 Saved reference face: {ref_save_path}")
                idx += 1

    if not embeddings:
        raise ValueError("❌ No valid reference embeddings found.")
    return np.mean(embeddings, axis=0)

# === VIDEO PROCESSING ===
def process_videos():
    ref_folder = reference_folder.get()
    vid_folder = video_folder.get()
    matched_folder = matched_videos_folder.get()
    ref_output = os.path.join(output_folder.get(), "reference_faces")
    skipped_folder = os.path.join(output_folder.get(), "skipped_faces")
    matched_faces_folder = os.path.join(output_folder.get(), "matched_faces")
    not_matched_folder = os.path.join(output_folder.get(), "not_matched_faces")

    threshold = float(distance_threshold.get())
    frame_jump_val = int(frame_jump.get())
    max_frames_val = int(max_frames.get())

    # Create required folders
    for folder in [ref_output, skipped_folder, matched_faces_folder, not_matched_folder, matched_folder]:
        os.makedirs(folder, exist_ok=True)

    # Load reference embeddings
    ref_embedding = load_reference_embeddings(ref_folder, ref_output)

    # Get video list
    video_files = [f for f in os.listdir(vid_folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    total_videos = len(video_files)
    progress_var.set(0)

    for i, video_name in enumerate(video_files, start=1):
        video_path = os.path.join(vid_folder, video_name)
        print(f"\n🎥 Processing video: {video_name}")
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
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                cropped_face = frame[y1:y2, x1:x2]

                if cropped_face.size == 0:
                    continue

                if cropped_face.shape[0] < 50 or cropped_face.shape[1] < 50 or is_blurry(cropped_face):
                    cv2.imwrite(os.path.join(skipped_folder, f"{video_name}_frame{frame_count:04d}_face{idx}.jpg"), cropped_face)
                    continue

                emb = face.normed_embedding
                distance_val = np.linalg.norm(emb - ref_embedding)

                if distance_val < threshold:
                    cv2.imwrite(os.path.join(matched_faces_folder, f"{video_name}_frame{frame_count:04d}_face{idx}_match.jpg"), cropped_face)
                    match_found = True
                    break
                else:
                    cv2.imwrite(os.path.join(not_matched_folder, f"{video_name}_frame{frame_count:04d}_face{idx}.jpg"), cropped_face)

            if match_found:
                break

        cap.release()

        if match_found:
            shutil.move(video_path, os.path.join(matched_folder, video_name))

        progress_var.set((i / total_videos) * 100)
        root.update_idletasks()

    print("🏁 All videos processed.")

def start_processing():
    thread = threading.Thread(target=process_videos)
    thread.start()

# === GUI ===
root = Tk()
root.title("Face Match Video Search")
root.geometry("700x500")

# Variables
reference_folder = StringVar()
video_folder = StringVar()
matched_videos_folder = StringVar()
output_folder = StringVar()
distance_threshold = StringVar(value="0.8")
frame_jump = StringVar(value="10")
max_frames = StringVar(value="5000")
progress_var = DoubleVar()

# Layout
Label(root, text="Reference Images Folder:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
Entry(root, textvariable=reference_folder, width=50).grid(row=0, column=1, padx=5, pady=5)
Button(root, text="Browse", command=lambda: reference_folder.set(filedialog.askdirectory())).grid(row=0, column=2, padx=5, pady=5)

Label(root, text="Videos Folder:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
Entry(root, textvariable=video_folder, width=50).grid(row=1, column=1, padx=5, pady=5)
Button(root, text="Browse", command=lambda: video_folder.set(filedialog.askdirectory())).grid(row=1, column=2, padx=5, pady=5)

Label(root, text="Matched Videos Output Folder:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
Entry(root, textvariable=matched_videos_folder, width=50).grid(row=2, column=1, padx=5, pady=5)
Button(root, text="Browse", command=lambda: matched_videos_folder.set(filedialog.askdirectory())).grid(row=2, column=2, padx=5, pady=5)

Label(root, text="Images Output Base Folder:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
Entry(root, textvariable=output_folder, width=50).grid(row=3, column=1, padx=5, pady=5)
Button(root, text="Browse", command=lambda: output_folder.set(filedialog.askdirectory())).grid(row=3, column=2, padx=5, pady=5)

Label(root, text="Distance Threshold:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
Entry(root, textvariable=distance_threshold).grid(row=4, column=1, padx=5, pady=5)

Label(root, text="Frame Jump:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
Entry(root, textvariable=frame_jump).grid(row=5, column=1, padx=5, pady=5)

Label(root, text="Max Frames:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
Entry(root, textvariable=max_frames).grid(row=6, column=1, padx=5, pady=5)

Button(root, text="Start Processing", command=start_processing).grid(row=7, column=0, columnspan=3, pady=10)

# Progress bar
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.grid(row=8, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

# Make resizing nice
for i in range(3):
    root.grid_columnconfigure(i, weight=1)

root.mainloop()
