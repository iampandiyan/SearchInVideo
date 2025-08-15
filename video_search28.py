import cv2
import numpy as np
import os
import shutil
import threading
from tkinter import Tk, Label, Entry, Button, filedialog, StringVar, DoubleVar, IntVar
from tkinter import ttk
from insightface.app import FaceAnalysis

#210 results with threshold 0.9  and fram 500 and result console
# === INIT INSIGHTFACE ===
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# === GLOBAL STOP FLAG ===
stop_flag = False

# === BLUR DETECTION ===
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

# === FACE PREPROCESSING ===
def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    img_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp_img = cv2.filter2D(img_eq, -1, kernel)
    return sharp_img

# === LOAD REFERENCE EMBEDDINGS ===
def load_reference_embeddings(folder_path, ref_output):
    embeddings = []
    idx = 0
    os.makedirs(ref_output, exist_ok=True)

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
    global stop_flag
    stop_flag = False  # Reset flag before starting

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

    for folder in [ref_output, skipped_folder, matched_faces_folder, not_matched_folder, matched_folder]:
        os.makedirs(folder, exist_ok=True)

    ref_embedding = load_reference_embeddings(ref_folder, ref_output)
    video_files = [f for f in os.listdir(vid_folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    
    total_videos = len(video_files)
    total_videos_var.set(total_videos)
    processed_videos_var.set(0)
    matched_videos_var.set(0)
    not_matched_videos_var.set(0)
    yet_to_process_var.set(total_videos)

    progress_var.set(0)

    for i, video_name in enumerate(video_files, start=1):
        if stop_flag:
            print("🛑 Processing stopped by user.")
            break

        video_path = os.path.join(vid_folder, video_name)
        print(f"\n🎥 Processing video: {video_name}")
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        match_found = False

        while cap.isOpened() and frame_count < max_frames_val:
            if stop_flag:
                print("🛑 Stopping mid-video.")
                break

            ret, frame = cap.read()
            if not ret:
                break
            frame_count += frame_jump_val
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            faces = app.get(frame)

            for idx, face in enumerate(faces):
                if stop_flag:
                    break

                x1, y1, x2, y2 = map(int, face.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                cropped_face = frame[y1:y2, x1:x2]

                if cropped_face.size == 0:
                    continue

                if cropped_face.shape[0] < 50 or cropped_face.shape[1] < 50 or is_blurry(cropped_face):
                    processed_face_retry = preprocess_face(cropped_face)
                    if processed_face_retry.shape[0] >= 50 and processed_face_retry.shape[1] >= 50 and not is_blurry(processed_face_retry):
                        cropped_face = processed_face_retry
                    else:
                        cv2.imwrite(os.path.join(skipped_folder, f"{video_name}_frame{frame_count:04d}_face{idx}.jpg"), cropped_face)
                        continue

                emb = face.normed_embedding
                distance_val = np.linalg.norm(emb - ref_embedding)

                if distance_val < threshold:
                    cv2.imwrite(os.path.join(matched_faces_folder, f"{video_name}_frame{frame_count:04d}_face{idx}_match.jpg"), cropped_face)
                    match_found = True
                    break
                else:
                    processed_face = preprocess_face(cropped_face)
                    processed_faces = app.get(processed_face)
                    if processed_faces:
                        processed_emb = processed_faces[0].normed_embedding
                        processed_distance = np.linalg.norm(processed_emb - ref_embedding)
                        if processed_distance < threshold:
                            cv2.imwrite(os.path.join(matched_faces_folder, f"{video_name}_frame{frame_count:04d}_face{idx}_match_fallback.jpg"), processed_face)
                            match_found = True
                            break
                    cv2.imwrite(os.path.join(not_matched_folder, f"{video_name}_frame{frame_count:04d}_face{idx}.jpg"), cropped_face)

            if match_found or stop_flag:
                break

        cap.release()

        # Update counters
        processed_videos_var.set(processed_videos_var.get() + 1)
        yet_to_process_var.set(total_videos - processed_videos_var.get())
        if match_found:
            matched_videos_var.set(matched_videos_var.get() + 1)
            shutil.move(video_path, os.path.join(matched_folder, video_name))
        else:
            not_matched_videos_var.set(not_matched_videos_var.get() + 1)

        progress_var.set((i / total_videos) * 100)
        root.update_idletasks()

    print("🏁 Processing finished or stopped.")

def start_processing():
    thread = threading.Thread(target=process_videos)
    thread.start()

def stop_processing():
    global stop_flag
    stop_flag = True

# === GUI ===
root = Tk()
root.title("Face Match Video Search")
root.geometry("750x600")

reference_folder = StringVar()
video_folder = StringVar()
matched_videos_folder = StringVar()
output_folder = StringVar()
distance_threshold = StringVar(value="0.8")
frame_jump = StringVar(value="10")
max_frames = StringVar(value="5000")
progress_var = DoubleVar()

# Stats variables
total_videos_var = IntVar(value=0)
processed_videos_var = IntVar(value=0)
matched_videos_var = IntVar(value=0)
not_matched_videos_var = IntVar(value=0)
yet_to_process_var = IntVar(value=0)

# Input fields
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

# Buttons
Button(root, text="Start Processing", command=start_processing).grid(row=7, column=0, columnspan=2, pady=10)
Button(root, text="Stop Processing", command=stop_processing).grid(row=7, column=2, pady=10)

# Progress Bar
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.grid(row=8, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

# Stats display
Label(root, text="Total Videos:").grid(row=9, column=0, sticky="w", padx=5)
Label(root, textvariable=total_videos_var).grid(row=9, column=1, sticky="w")

Label(root, text="Videos Processed:").grid(row=10, column=0, sticky="w", padx=5)
Label(root, textvariable=processed_videos_var).grid(row=10, column=1, sticky="w")

Label(root, text="Videos Matched:").grid(row=11, column=0, sticky="w", padx=5)
Label(root, textvariable=matched_videos_var).grid(row=11, column=1, sticky="w")

Label(root, text="Videos Not Matched:").grid(row=12, column=0, sticky="w", padx=5)
Label(root, textvariable=not_matched_videos_var).grid(row=12, column=1, sticky="w")

Label(root, text="Videos Yet to Process:").grid(row=13, column=0, sticky="w", padx=5)
Label(root, textvariable=yet_to_process_var).grid(row=13, column=1, sticky="w")

for i in range(3):
    root.grid_columnconfigure(i, weight=1)

root.mainloop()
