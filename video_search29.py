import cv2
import numpy as np
import os
import shutil
import threading
from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis

# === INIT INSIGHTFACE ===
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

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
    return cv2.filter2D(img_eq, -1, kernel)

# === LOAD REFERENCE EMBEDDINGS ===
def load_reference_embeddings(folder_path, ref_output):
    embeddings = []
    os.makedirs(ref_output, exist_ok=True)

    for source in [folder_path, os.path.join(folder_path, "manually_selected_faces")]:
        if not os.path.exists(source):
            continue
        for filename in os.listdir(source):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                img = cv2.imread(os.path.join(source, filename))
                if img is None:
                    continue
                faces = app.get(img)
                if not faces:
                    continue
                embeddings.append(faces[0].normed_embedding)

    if not embeddings:
        raise ValueError("❌ No valid reference embeddings found.")
    return np.mean(embeddings, axis=0)

# === Populate Faces in Tabs ===
def load_faces_to_tab(folder, container):
    for widget in container.winfo_children():
        widget.destroy()
    row, col = 0, 0
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder, file)
            img = Image.open(img_path).resize((100, 100))
            img_tk = ImageTk.PhotoImage(img)
            lbl = Label(container, image=img_tk)
            lbl.image = img_tk
            lbl.grid(row=row, column=col, padx=2, pady=2)
            col += 1
            if col >= 6:
                col = 0
                row += 1

# === VIDEO PROCESSING ===
def process_videos():
    global stop_flag
    stop_flag = False

    ref_embedding = load_reference_embeddings(reference_folder.get(), os.path.join(output_folder.get(), "reference_faces"))

    matched_faces_folder = os.path.join(output_folder.get(), "matched_faces")
    not_matched_folder = os.path.join(output_folder.get(), "not_matched_faces")
    skipped_folder = os.path.join(output_folder.get(), "skipped_faces")
    matched_videos_out = matched_videos_folder.get()
    os.makedirs(matched_faces_folder, exist_ok=True)
    os.makedirs(not_matched_folder, exist_ok=True)
    os.makedirs(skipped_folder, exist_ok=True)

    video_files = [f for f in os.listdir(video_folder.get()) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    total_videos_var.set(len(video_files))
    processed_videos_var.set(0)
    matched_videos_var.set(0)
    not_matched_videos_var.set(0)
    yet_to_process_var.set(len(video_files))
    progress_var.set(0)

    for i, video_name in enumerate(video_files, start=1):
        if stop_flag:
            break

        video_path = os.path.join(video_folder.get(), video_name)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        match_found = False

        while cap.isOpened() and frame_count < int(max_frames.get()):
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += int(frame_jump.get())
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            faces = app.get(frame)

            for idx, face in enumerate(faces):
                x1, y1, x2, y2 = map(int, face.bbox)
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                if is_blurry(cropped):
                    cropped = preprocess_face(cropped)
                if cropped.shape[0] < 50 or cropped.shape[1] < 50:
                    cv2.imwrite(os.path.join(skipped_folder, f"{video_name}_f{frame_count}_face{idx}.jpg"), cropped)
                    continue
                emb = face.normed_embedding
                dist = np.linalg.norm(emb - ref_embedding)
                if dist < float(distance_threshold.get()):
                    cv2.imwrite(os.path.join(matched_faces_folder, f"{video_name}_f{frame_count}_match.jpg"), cropped)
                    match_found = True
                    break
                else:
                    cv2.imwrite(os.path.join(not_matched_folder, f"{video_name}_f{frame_count}_nomatch.jpg"), cropped)

            if match_found:
                break
        cap.release()

        processed_videos_var.set(processed_videos_var.get() + 1)
        yet_to_process_var.set(total_videos_var.get() - processed_videos_var.get())
        if match_found:
            matched_videos_var.set(matched_videos_var.get() + 1)
            shutil.move(video_path, os.path.join(matched_videos_out, video_name))
        else:
            not_matched_videos_var.set(not_matched_videos_var.get() + 1)

        progress_var.set((i / total_videos_var.get()) * 100)

    # Populate tabs after processing finishes
    load_faces_to_tab(matched_faces_folder, matched_faces_frame_inner)
    load_faces_to_tab(not_matched_folder, unmatched_faces_frame_inner)
    load_faces_to_tab(skipped_folder, skipped_faces_frame_inner)

def start_processing():
    threading.Thread(target=process_videos).start()

def stop_processing():
    global stop_flag
    stop_flag = True

def reset_all():
    for var in [reference_folder, video_folder, matched_videos_folder, output_folder]:
        var.set("")
    for var in [total_videos_var, processed_videos_var, matched_videos_var, not_matched_videos_var, yet_to_process_var]:
        var.set(0)
    progress_var.set(0)
    for container in [matched_faces_frame_inner, unmatched_faces_frame_inner, skipped_faces_frame_inner]:
        for widget in container.winfo_children():
            widget.destroy()

# === GUI ===
root = Tk()
root.title("Face Match Video Search")
root.geometry("950x700")

reference_folder = StringVar()
video_folder = StringVar()
matched_videos_folder = StringVar()
output_folder = StringVar()
distance_threshold = StringVar(value="0.8")
frame_jump = StringVar(value="10")
max_frames = StringVar(value="5000")
progress_var = DoubleVar()
total_videos_var = IntVar()
processed_videos_var = IntVar()
matched_videos_var = IntVar()
not_matched_videos_var = IntVar()
yet_to_process_var = IntVar()

notebook = ttk.Notebook(root)
tab1 = Frame(notebook)
tab2 = Frame(notebook)
tab3 = Frame(notebook)
tab4 = Frame(notebook)
notebook.add(tab1, text="Stats")
notebook.add(tab2, text="Matched Faces")
notebook.add(tab3, text="Unmatched Faces")
notebook.add(tab4, text="Skipped Faces")
notebook.pack(expand=True, fill="both")

# Tab 1 (Stats)
Label(tab1, text="Reference Images Folder:").grid(row=0, column=0, sticky="w")
Entry(tab1, textvariable=reference_folder, width=50).grid(row=0, column=1)
Button(tab1, text="Browse", command=lambda: reference_folder.set(filedialog.askdirectory())).grid(row=0, column=2)

Label(tab1, text="Videos Folder:").grid(row=1, column=0, sticky="w")
Entry(tab1, textvariable=video_folder, width=50).grid(row=1, column=1)
Button(tab1, text="Browse", command=lambda: video_folder.set(filedialog.askdirectory())).grid(row=1, column=2)

Label(tab1, text="Matched Videos Folder:").grid(row=2, column=0, sticky="w")
Entry(tab1, textvariable=matched_videos_folder, width=50).grid(row=2, column=1)
Button(tab1, text="Browse", command=lambda: matched_videos_folder.set(filedialog.askdirectory())).grid(row=2, column=2)

Label(tab1, text="Images Output Base Folder:").grid(row=3, column=0, sticky="w")
Entry(tab1, textvariable=output_folder, width=50).grid(row=3, column=1)
Button(tab1, text="Browse", command=lambda: output_folder.set(filedialog.askdirectory())).grid(row=3, column=2)

Label(tab1, text="Distance Threshold:").grid(row=4, column=0, sticky="w")
Entry(tab1, textvariable=distance_threshold).grid(row=4, column=1)

Label(tab1, text="Frame Jump:").grid(row=5, column=0, sticky="w")
Entry(tab1, textvariable=frame_jump).grid(row=5, column=1)

Label(tab1, text="Max Frames:").grid(row=6, column=0, sticky="w")
Entry(tab1, textvariable=max_frames).grid(row=6, column=1)

Button(tab1, text="Start Processing", command=start_processing).grid(row=7, column=0, pady=10)
Button(tab1, text="Stop Processing", command=stop_processing).grid(row=7, column=1, pady=10)
Button(tab1, text="Reset", command=reset_all).grid(row=7, column=2, pady=10)

ttk.Progressbar(tab1, variable=progress_var, maximum=100).grid(row=8, column=0, columnspan=3, sticky="ew", pady=5)

Label(tab1, text="Total Videos:").grid(row=9, column=0, sticky="w")
Label(tab1, textvariable=total_videos_var).grid(row=9, column=1, sticky="w")
Label(tab1, text="Processed:").grid(row=10, column=0, sticky="w")
Label(tab1, textvariable=processed_videos_var).grid(row=10, column=1, sticky="w")
Label(tab1, text="Matched:").grid(row=11, column=0, sticky="w")
Label(tab1, textvariable=matched_videos_var).grid(row=11, column=1, sticky="w")
Label(tab1, text="Not Matched:").grid(row=12, column=0, sticky="w")
Label(tab1, textvariable=not_matched_videos_var).grid(row=12, column=1, sticky="w")
Label(tab1, text="Yet to Process:").grid(row=13, column=0, sticky="w")
Label(tab1, textvariable=yet_to_process_var).grid(row=13, column=1, sticky="w")

# Tab 2-4 (Face Display)
def create_scrollable_tab(tab):
    canvas = Canvas(tab)
    scrollbar = Scrollbar(tab, orient="vertical", command=canvas.yview)
    scroll_frame = Frame(canvas)
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return scroll_frame

matched_faces_frame_inner = create_scrollable_tab(tab2)
unmatched_faces_frame_inner = create_scrollable_tab(tab3)
skipped_faces_frame_inner = create_scrollable_tab(tab4)

root.mainloop()
