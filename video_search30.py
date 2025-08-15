import cv2
import numpy as np
import os
import shutil
import threading
import traceback
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis

# ========= InsightFace =========
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ========= Globals / Flags =========
stop_flag = False
processing_thread = None
processing_lock = threading.Lock()
ref_embeddings = []  # we will keep multiple embeddings; match by min distance

# ========= Helpers =========
def log(msg: str):
    """Print to terminal + append to the GUI console."""
    print(msg, flush=True)
    if console_text is not None:
        console_text.configure(state="normal")
        console_text.insert(END, msg + "\n")
        console_text.see(END)
        console_text.configure(state="disabled")

def safe_after(ms, fn, *args, **kwargs):
    """Schedule UI updates from worker thread."""
    try:
        root.after(ms, fn, *args, **kwargs)
    except Exception:
        # In case root is already destroyed
        pass

# ========= Image Quality =========
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    img_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img_eq, -1, kernel)

# ========= Reference Handling =========
def load_reference_embeddings(folder_path, ref_output):
    """
    Load all face embeddings from:
      - folder_path
      - folder_path/manually_selected_faces (if exists)
    Returns: list of embeddings (each is a vector)
    """
    embeddings = []
    os.makedirs(ref_output, exist_ok=True)

    for source in [folder_path, os.path.join(folder_path, "manually_selected_faces")]:
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
                    continue
                emb = faces[0].normed_embedding
                embeddings.append(emb)

    if not embeddings:
        raise ValueError("❌ No valid reference embeddings found in the selected folder(s).")
    return embeddings

def get_min_distance(emb, ref_list):
    """Return min L2 distance between emb and any embedding in ref_list."""
    if not ref_list:
        return float("inf")
    # stack for vectorized distance
    refs = np.stack(ref_list, axis=0)  # [N, D]
    diffs = refs - emb
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    return float(np.min(dists))

# ========= Thumbnails / Tabs =========
def clear_container(container):
    for w in container.winfo_children():
        w.destroy()

def open_preview(img_path):
    """Popup window to preview a face image."""
    try:
        win = Toplevel(root)
        win.title(os.path.basename(img_path))
        img = Image.open(img_path)
        # limit size for display
        max_side = 640
        ratio = min(max_side / img.width, max_side / img.height, 1.0)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        tkimg = ImageTk.PhotoImage(img)
        lbl = Label(win, image=tkimg)
        lbl.image = tkimg
        lbl.pack(padx=10, pady=10)
    except Exception as e:
        messagebox.showerror("Preview error", str(e))

def load_faces_to_tab(folder, container):
    """Fill a tab with thumbnails from folder; click to preview."""
    clear_container(container)
    if not os.path.exists(folder):
        return
    row, col = 0, 0
    for file in sorted(os.listdir(folder)):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder, file)
            try:
                img = Image.open(img_path).resize((110, 110))
                img_tk = ImageTk.PhotoImage(img)
                btn = Button(container, image=img_tk, command=lambda p=img_path: open_preview(p))
                btn.image = img_tk  # keep reference
                btn.grid(row=row, column=col, padx=4, pady=4)
                col += 1
                if col >= 6:
                    col = 0
                    row += 1
            except Exception:
                continue

# ========= Video Processing =========
def process_videos():
    global stop_flag, ref_embeddings

    try:
        with processing_lock:
            stop_flag = False

        # Disable/Enable buttons (UI) at start
        safe_after(0, start_btn.config, {"state": DISABLED})
        safe_after(0, stop_btn.config, {"state": NORMAL})

        # Validate inputs
        if not reference_folder.get() or not os.path.isdir(reference_folder.get()):
            raise ValueError("Please select a valid Reference Images Folder.")
        if not video_folder.get() or not os.path.isdir(video_folder.get()):
            raise ValueError("Please select a valid Videos Folder.")
        if not matched_videos_folder.get() or not os.path.isdir(matched_videos_folder.get()):
            raise ValueError("Please select a valid Matched Videos Folder.")
        if not output_folder.get() or not os.path.isdir(output_folder.get()):
            raise ValueError("Please select a valid Images Output Base Folder.")

        # Load reference embeddings (list)
        ref_out_dir = os.path.join(output_folder.get(), "reference_faces")
        log("Loading reference embeddings...")
        ref_embeddings = load_reference_embeddings(reference_folder.get(), ref_out_dir)
        log(f"Loaded {len(ref_embeddings)} reference embeddings.")

        # Prepare output dirs
        matched_faces_folder = os.path.join(output_folder.get(), "matched_faces")
        not_matched_folder = os.path.join(output_folder.get(), "not_matched_faces")
        skipped_folder = os.path.join(output_folder.get(), "skipped_faces")
        os.makedirs(matched_faces_folder, exist_ok=True)
        os.makedirs(not_matched_folder, exist_ok=True)
        os.makedirs(skipped_folder, exist_ok=True)

        # Gather video files
        video_files = [f for f in os.listdir(video_folder.get())
                       if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        total = len(video_files)

        # Initialize stats
        safe_after(0, total_videos_var.set, total)
        safe_after(0, processed_videos_var.set, 0)
        safe_after(0, matched_videos_var.set, 0)
        safe_after(0, not_matched_videos_var.set, 0)
        safe_after(0, yet_to_process_var.set, total)
        safe_after(0, progress_var.set, 0)

        if total == 0:
            log("No videos found in the selected folder.")
            return

        # Parameters
        dist_thr = float(distance_threshold.get())
        frame_step = int(frame_jump.get())
        max_fr = int(max_frames.get())

        log(f"Starting processing of {total} video(s) with threshold={dist_thr}, frame_step={frame_step}, max_frames={max_fr}")

        # ---- Process each video ----
        for idx_video, video_name in enumerate(sorted(video_files), start=1):
            # Stop request?
            with processing_lock:
                if stop_flag:
                    log("🛑 Stop requested by user. Ending early.")
                    break

            video_path = os.path.join(video_folder.get(), video_name)
            log(f"\n🎥 [{idx_video}/{total}] Processing: {video_name}")

            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            match_found = False

            if not cap.isOpened():
                log(f"⚠️ Cannot open video: {video_name}")
                cap.release()
                # Update stats even if unreadable
                safe_after(0, processed_videos_var.set, processed_videos_var.get() + 1)
                pending = max(total - processed_videos_var.get(), 0)
                safe_after(0, yet_to_process_var.set, pending)
                safe_after(0, progress_var.set, (processed_videos_var.get() / max(total, 1)) * 100.0)
                continue

            while cap.isOpened() and frame_count < max_fr:
                with processing_lock:
                    if stop_flag:
                        log("🛑 Stop requested mid-video.")
                        break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += frame_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

                faces = app.get(frame)
                if not faces:
                    continue

                for fi, face in enumerate(faces):
                    x1, y1, x2, y2 = map(int, face.bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size == 0:
                        continue

                    # Try to improve quality if blurry/small
                    candidate = cropped
                    if candidate.shape[0] < 50 or candidate.shape[1] < 50 or is_blurry(candidate):
                        candidate = preprocess_face(candidate)

                    # If still too small after preprocessing, skip (save to skipped)
                    if candidate.shape[0] < 50 or candidate.shape[1] < 50:
                        outp = os.path.join(skipped_folder, f"{os.path.splitext(video_name)[0]}_f{frame_count:06d}_face{fi}.jpg")
                        cv2.imwrite(outp, candidate)
                        continue

                    emb = face.normed_embedding
                    dist = get_min_distance(emb, ref_embeddings)

                    if dist < dist_thr:
                        # Save match
                        outp = os.path.join(matched_faces_folder, f"{os.path.splitext(video_name)[0]}_f{frame_count:06d}_match.jpg")
                        cv2.imwrite(outp, candidate)
                        match_found = True

                        # Add embedding to references (dynamic growth)
                        ref_embeddings.append(emb)

                        log(f"✅ Match at frame {frame_count} (dist {dist:.4f}) → {os.path.basename(outp)}")
                        break
                    else:
                        # Save non-match sample
                        outp = os.path.join(not_matched_folder, f"{os.path.splitext(video_name)[0]}_f{frame_count:06d}_nomatch.jpg")
                        cv2.imwrite(outp, candidate)

                if match_found:
                    break

            cap.release()

            # Tally + move video if matched
            safe_after(0, processed_videos_var.set, processed_videos_var.get() + 1)
            if match_found:
                safe_after(0, matched_videos_var.set, matched_videos_var.get() + 1)
                try:
                    shutil.move(video_path, os.path.join(matched_videos_folder.get(), video_name))
                    log(f"📦 Moved matched video → {matched_videos_folder.get()}")
                except Exception as e:
                    log(f"⚠️ Could not move video: {e}")
            else:
                safe_after(0, not_matched_videos_var.set, not_matched_videos_var.get() + 1)
                log("❌ No match found in this video.")

            # Update remaining and progress safely
            processed_now = processed_videos_var.get()
            pending = max(total - processed_now, 0)
            safe_after(0, yet_to_process_var.set, pending)
            safe_after(0, progress_var.set, (processed_now / max(total, 1)) * 100.0)

        # After loop, fill tabs (only once processing finishes/stops)
        safe_after(0, load_faces_to_tab, os.path.join(output_folder.get(), "matched_faces"), matched_faces_frame_inner)
        safe_after(0, load_faces_to_tab, os.path.join(output_folder.get(), "not_matched_faces"), unmatched_faces_frame_inner)
        safe_after(0, load_faces_to_tab, os.path.join(output_folder.get(), "skipped_faces"), skipped_faces_frame_inner)

    except Exception as e:
        log("🔥 Error during processing:\n" + traceback.format_exc())
        messagebox.showerror("Processing Error", str(e))

    finally:
        # Re-enable/disable buttons at end
        safe_after(0, start_btn.config, {"state": NORMAL})
        safe_after(0, stop_btn.config, {"state": DISABLED})
        log("\n🏁 Done.\n")

# ========= Start/Stop/Reset =========
def start_processing():
    global processing_thread, stop_flag
    # Clear console & thumbnails (tabs empty at start)
    console_text.configure(state="normal")
    console_text.delete("1.0", END)
    console_text.configure(state="disabled")
    clear_container(matched_faces_frame_inner)
    clear_container(unmatched_faces_frame_inner)
    clear_container(skipped_faces_frame_inner)

    # Reset counters
    total_videos_var.set(0)
    processed_videos_var.set(0)
    matched_videos_var.set(0)
    not_matched_videos_var.set(0)
    yet_to_process_var.set(0)
    progress_var.set(0)

    with processing_lock:
        stop_flag = False

    # Disable start immediately to prevent double-runs
    start_btn.config(state=DISABLED)
    stop_btn.config(state=NORMAL)

    processing_thread = threading.Thread(target=process_videos, daemon=True)
    processing_thread.start()

def stop_processing():
    global stop_flag
    with processing_lock:
        stop_flag = True
    log("🛑 Stop requested. Finishing current step...")

def reset_all():
    # Only allow reset when not processing
    if processing_thread and processing_thread.is_alive():
        messagebox.showwarning("Busy", "Please wait until processing finishes or click Stop.")
        return

    for var in [reference_folder, video_folder, matched_videos_folder, output_folder]:
        var.set("")
    for var in [total_videos_var, processed_videos_var, matched_videos_var,
                not_matched_videos_var, yet_to_process_var]:
        var.set(0)
    progress_var.set(0)

    clear_container(matched_faces_frame_inner)
    clear_container(unmatched_faces_frame_inner)
    clear_container(skipped_faces_frame_inner)

    console_text.configure(state="normal")
    console_text.delete("1.0", END)
    console_text.configure(state="disabled")

# ========= GUI =========
root = Tk()
root.title("Face Match Video Search (InsightFace)")
root.geometry("1060x780")

reference_folder = StringVar()
video_folder = StringVar()
matched_videos_folder = StringVar()
output_folder = StringVar()
distance_threshold = StringVar(value="0.8")
frame_jump = StringVar(value="10")
max_frames = StringVar(value="5000")

progress_var = DoubleVar(value=0.0)
total_videos_var = IntVar(value=0)
processed_videos_var = IntVar(value=0)
matched_videos_var = IntVar(value=0)
not_matched_videos_var = IntVar(value=0)
yet_to_process_var = IntVar(value=0)

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

# ---- Tab 1: Stats / Controls ----
def rowpair(parent, r, label, var=None, width=50, is_entry=True, browse=False, cmd=None):
    Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
    if is_entry:
        Entry(parent, textvariable=var, width=width).grid(row=r, column=1, sticky="ew", padx=6)
    else:
        Label(parent, textvariable=var).grid(row=r, column=1, sticky="w", padx=6)
    if browse:
        Button(parent, text="Browse", command=cmd).grid(row=r, column=2, padx=6)

rowpair(tab1, 0, "Reference Images Folder:", reference_folder, browse=True,
        cmd=lambda: reference_folder.set(filedialog.askdirectory()))
rowpair(tab1, 1, "Videos Folder:", video_folder, browse=True,
        cmd=lambda: video_folder.set(filedialog.askdirectory()))
rowpair(tab1, 2, "Matched Videos Folder:", matched_videos_folder, browse=True,
        cmd=lambda: matched_videos_folder.set(filedialog.askdirectory()))
rowpair(tab1, 3, "Images Output Base Folder:", output_folder, browse=True,
        cmd=lambda: output_folder.set(filedialog.askdirectory()))

rowpair(tab1, 4, "Distance Threshold:", distance_threshold)
rowpair(tab1, 5, "Frame Jump:", frame_jump)
rowpair(tab1, 6, "Max Frames:", max_frames)

buttons_frame = Frame(tab1)
buttons_frame.grid(row=7, column=0, columnspan=3, pady=10, sticky="w")

start_btn = Button(buttons_frame, text="Start Processing", command=start_processing, width=18)
stop_btn = Button(buttons_frame, text="Stop Processing", command=stop_processing, width=16, state=DISABLED)
reset_btn = Button(buttons_frame, text="Reset", command=reset_all, width=10)

start_btn.grid(row=0, column=0, padx=4)
stop_btn.grid(row=0, column=1, padx=4)
reset_btn.grid(row=0, column=2, padx=4)

ttk.Progressbar(tab1, variable=progress_var, maximum=100).grid(row=8, column=0, columnspan=3, sticky="ew", padx=6, pady=4)

rowpair(tab1, 9,  "Total Videos:", total_videos_var, is_entry=False)
rowpair(tab1, 10, "Processed:", processed_videos_var, is_entry=False)
rowpair(tab1, 11, "Matched:", matched_videos_var, is_entry=False)
rowpair(tab1, 12, "Not Matched:", not_matched_videos_var, is_entry=False)
rowpair(tab1, 13, "Yet to Process:", yet_to_process_var, is_entry=False)

# Console log box
Label(tab1, text="Console Log:").grid(row=14, column=0, sticky="w", padx=6, pady=(8, 2))
console_text = scrolledtext.ScrolledText(tab1, height=10, state="disabled", wrap="word")
console_text.grid(row=15, column=0, columnspan=3, sticky="nsew", padx=6, pady=(0, 8))

tab1.grid_columnconfigure(1, weight=1)
tab1.grid_rowconfigure(15, weight=1)

# ---- Tabs 2-4: scrollable thumbnails ----
def create_scrollable_tab(tab):
    canvas = Canvas(tab)
    scrollbar = Scrollbar(tab, orient="vertical", command=canvas.yview)
    scroll_frame = Frame(canvas)

    def on_configure(e):
        canvas.configure(scrollregion=canvas.bbox("all"))

    scroll_frame.bind("<Configure>", on_configure)
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return scroll_frame

matched_faces_frame_inner = create_scrollable_tab(tab2)
unmatched_faces_frame_inner = create_scrollable_tab(tab3)
skipped_faces_frame_inner = create_scrollable_tab(tab4)

# Start with empty tabs (they’ll fill after processing completes)
clear_container(matched_faces_frame_inner)
clear_container(unmatched_faces_frame_inner)
clear_container(skipped_faces_frame_inner)

root.mainloop()
