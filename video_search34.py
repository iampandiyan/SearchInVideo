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

#Perfectly working with Move option
# ========= InsightFace =========
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ========= Globals / Flags =========
stop_flag = False
processing_thread = None
processing_lock = threading.Lock()
ref_embeddings = []  # we will keep multiple embeddings; match by min distance
selected_images = set() # To store paths of selected images for moving

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
    Load all face embeddings and save cropped faces.
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
                
                # Crop and save the reference face
                face = faces[0] # Assuming one face per reference image
                x1, y1, x2, y2 = map(int, face.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                cropped_face = img[y1:y2, x1:x2]
                if cropped_face.size > 0:
                    # Save the cropped face to the designated reference_faces output folder
                    save_path = os.path.join(ref_output, f"ref_{filename}")
                    cv2.imwrite(save_path, cropped_face)

                emb = face.normed_embedding
                embeddings.append(emb)

    if not embeddings:
        raise ValueError("❌ No valid reference embeddings found in the selected folder(s).")
    return embeddings

def get_min_distance(emb, ref_list):
    """Return min L2 distance between emb and any embedding in ref_list."""
    if not ref_list:
        return float("inf")
    refs = np.stack(ref_list, axis=0)  # [N, D]
    diffs = refs - emb
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    return float(np.min(dists))

# ========= Thumbnails / Tabs =========
def get_video_base_name(image_path):
    """Extracts video base name from image filename like 'my_video_f123_...jpg'"""
    filename = os.path.basename(image_path)
    # Assumes filename format is VIDEO-BASENAME_f...
    parts = filename.split('_f')
    if len(parts) > 1:
        return parts[0]
    return None

def move_selected_files():
    """Moves selected videos and their associated images."""
    if not selected_images:
        messagebox.showinfo("Info", "No images selected.")
        return

    videos_to_process = set()
    for img_path in selected_images:
        base_name = get_video_base_name(img_path)
        if base_name:
            videos_to_process.add(base_name)

    log(f"Move operation started for {len(videos_to_process)} video(s).")

    src_video_dir = video_folder.get()
    dest_video_dir = matched_videos_folder.get()
    unmatched_img_dir = os.path.join(output_folder.get(), "not_matched_faces")
    skipped_img_dir = os.path.join(output_folder.get(), "skipped_faces")
    moved_img_dir = os.path.join(output_folder.get(), "moved_faces")

    if not all([src_video_dir, dest_video_dir, output_folder.get()]):
         messagebox.showerror("Error", "One or more source/destination folders are not set.")
         return

    moved_count = 0
    for base_name in videos_to_process:
        # 1. Find and move the video file
        original_video_path = None
        for f in os.listdir(src_video_dir):
            if os.path.splitext(f)[0] == base_name:
                original_video_path = os.path.join(src_video_dir, f)
                break
        
        if original_video_path and os.path.exists(original_video_path):
            try:
                shutil.move(original_video_path, dest_video_dir)
                log(f"📦 Moved video: {os.path.basename(original_video_path)} -> {dest_video_dir}")
                moved_count += 1
            except Exception as e:
                log(f"⚠️ Error moving video {os.path.basename(original_video_path)}: {e}")
        else:
            log(f"ℹ️ Video for '{base_name}' not found in source folder (might be already moved).")

        # 2. Find and move all related images from unmatched and skipped folders
        for folder in [unmatched_img_dir, skipped_img_dir]:
            if not os.path.exists(folder): continue
            for img_file in os.listdir(folder):
                if img_file.startswith(base_name + '_'):
                    src_path = os.path.join(folder, img_file)
                    dest_path = os.path.join(moved_img_dir, img_file)
                    try:
                        shutil.move(src_path, dest_path)
                    except Exception as e:
                        log(f"⚠️ Error moving image {img_file}: {e}")

    log(f"Move operation complete. Moved {moved_count} video file(s) and associated images.")
    
    selected_images.clear()
    refresh_face_tabs()

def toggle_selection(widget, img_path):
    """Add/remove image from selection and update widget appearance."""
    if img_path in selected_images:
        selected_images.remove(img_path)
        widget.config(relief="flat", borderwidth=1, bg=root.cget('bg'))
    else:
        selected_images.add(img_path)
        widget.config(relief="solid", borderwidth=2, bg="#ADD8E6") # Light blue highlight

def show_popup_menu(event, img_path, widget):
    """Show the right-click menu and toggle selection."""
    toggle_selection(widget, img_path)
    popup_menu.post(event.x_root, event.y_root)

def clear_container(container):
    for w in container.winfo_children():
        w.destroy()

def open_preview(img_path):
    """Popup window to preview a face image."""
    try:
        win = Toplevel(root)
        win.title(os.path.basename(img_path))
        img = Image.open(img_path).convert("RGB")
        max_side = 640
        ratio = min(max_side / img.width, max_side / img.height, 1.0)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        tkimg = ImageTk.PhotoImage(img)
        lbl = Label(win, image=tkimg)
        lbl.image = tkimg
        lbl.pack(padx=10, pady=10)
    except Exception as e:
        messagebox.showerror("Preview error", str(e))

def load_faces_to_tab(folder, container, tab_name=None):
    """Fill a tab with thumbnails; click to preview."""
    clear_container(container)
    if not os.path.exists(folder):
        return
    row, col = 0, 0
    for file in sorted(os.listdir(folder)):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder, file)
            try:
                img = Image.open(img_path).convert("RGB").resize((110, 110))
                img_tk = ImageTk.PhotoImage(img)
                
                btn = Button(container, image=img_tk, relief="flat", borderwidth=1)
                btn.image = img_tk
                
                # Add right-click context menu for specific tabs
                if tab_name in ["unmatched", "skipped"]:
                    btn.config(command=lambda p=img_path: open_preview(p))
                    btn.bind("<Button-3>", lambda e, p=img_path, w=btn: show_popup_menu(e, p, w))
                else:
                    btn.config(command=lambda p=img_path: open_preview(p))
                
                btn.grid(row=row, column=col, padx=4, pady=4)

                # Re-apply selection visual if image is already selected
                if img_path in selected_images:
                    btn.config(relief="solid", borderwidth=2, bg="#ADD8E6")
                
                col += 1
                if col >= 6:
                    col = 0
                    row += 1
            except Exception:
                continue
    container.update_idletasks()

def refresh_face_tabs():
    """Refresh all face tabs from the current output folders."""
    ref_faces_folder = os.path.join(output_folder.get(), "reference_faces")
    matched_faces_folder = os.path.join(output_folder.get(), "matched_faces")
    not_matched_folder = os.path.join(output_folder.get(), "not_matched_faces")
    skipped_folder = os.path.join(output_folder.get(), "skipped_faces")
    moved_faces_folder = os.path.join(output_folder.get(), "moved_faces")
    
    load_faces_to_tab(ref_faces_folder, reference_faces_frame_inner)
    load_faces_to_tab(matched_faces_folder, matched_faces_frame_inner)
    load_faces_to_tab(not_matched_folder, unmatched_faces_frame_inner, tab_name="unmatched")
    load_faces_to_tab(skipped_folder, skipped_faces_frame_inner, tab_name="skipped")
    load_faces_to_tab(moved_faces_folder, moved_faces_frame_inner)

# ========= Video Processing =========
def process_videos():
    global stop_flag, ref_embeddings

    try:
        with processing_lock:
            stop_flag = False

        safe_after(0, start_btn.config, {"state": DISABLED})
        safe_after(0, stop_btn.config, {"state": NORMAL})

        if not reference_folder.get() or not os.path.isdir(reference_folder.get()):
            raise ValueError("Please select a valid Reference Images Folder.")
        if not video_folder.get() or not os.path.isdir(video_folder.get()):
            raise ValueError("Please select a valid Videos Folder.")
        if not matched_videos_folder.get() or not os.path.isdir(matched_videos_folder.get()):
            raise ValueError("Please select a valid Matched Videos Folder.")
        if not output_folder.get() or not os.path.isdir(output_folder.get()):
            raise ValueError("Please select a valid Images Output Base Folder.")

        ref_out_dir = os.path.join(output_folder.get(), "reference_faces")
        log("Loading reference embeddings...")
        ref_embeddings = load_reference_embeddings(reference_folder.get(), ref_out_dir)
        log(f"Loaded {len(ref_embeddings)} reference embeddings.")

        # Create all output directories
        matched_faces_folder = os.path.join(output_folder.get(), "matched_faces")
        not_matched_folder = os.path.join(output_folder.get(), "not_matched_faces")
        skipped_folder = os.path.join(output_folder.get(), "skipped_faces")
        moved_faces_folder = os.path.join(output_folder.get(), "moved_faces")
        os.makedirs(matched_faces_folder, exist_ok=True)
        os.makedirs(not_matched_folder, exist_ok=True)
        os.makedirs(skipped_folder, exist_ok=True)
        os.makedirs(moved_faces_folder, exist_ok=True)

        video_files = [f for f in os.listdir(video_folder.get())
                       if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        total = len(video_files)

        safe_after(0, total_videos_var.set, total)
        safe_after(0, processed_videos_var.set, 0)
        safe_after(0, matched_videos_var.set, 0)
        safe_after(0, not_matched_videos_var.set, 0)
        safe_after(0, yet_to_process_var.set, total)
        safe_after(0, progress_var.set, 0)

        if total == 0:
            log("No videos found in the selected folder.")
            return

        dist_thr = float(distance_threshold.get())
        frame_step = int(frame_jump.get())
        max_fr = int(max_frames.get())

        log(f"Starting processing of {total} video(s) with threshold={dist_thr}, frame_step={frame_step}, max_frames={max_fr}")

        for idx_video, video_name in enumerate(sorted(video_files), start=1):
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
                safe_after(0, processed_videos_var.set, processed_videos_var.get() + 1)
                pending = max(total - processed_videos_var.get(), 0)
                safe_after(0, yet_to_process_var.set, pending)
                safe_after(0, progress_var.set, (processed_videos_var.get() / max(total, 1)) * 100.0)
                safe_after(0, refresh_face_tabs)
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

                    candidate = cropped
                    if candidate.shape[0] < 50 or candidate.shape[1] < 50 or is_blurry(candidate):
                        candidate = preprocess_face(candidate)

                    if candidate.shape[0] < 50 or candidate.shape[1] < 50:
                        outp = os.path.join(skipped_folder, f"{os.path.splitext(video_name)[0]}_f{frame_count:06d}_face{fi}.jpg")
                        cv2.imwrite(outp, candidate)
                        continue

                    emb = face.normed_embedding
                    dist = get_min_distance(emb, ref_embeddings)

                    if dist < dist_thr:
                        outp = os.path.join(matched_faces_folder, f"{os.path.splitext(video_name)[0]}_f{frame_count:06d}_match.jpg")
                        cv2.imwrite(outp, candidate)
                        match_found = True
                        ref_embeddings.append(emb)
                        log(f"✅ Match at frame {frame_count} (dist {dist:.4f}) → {os.path.basename(outp)}")
                        break
                    else:
                        outp = os.path.join(not_matched_folder, f"{os.path.splitext(video_name)[0]}_f{frame_count:06d}_nomatch.jpg")
                        cv2.imwrite(outp, candidate)

                if match_found:
                    break

            cap.release()

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

            processed_now = processed_videos_var.get()
            pending = max(total - processed_now, 0)
            safe_after(0, yet_to_process_var.set, pending)
            safe_after(0, progress_var.set, (processed_now / max(total, 1)) * 100.0)
            safe_after(0, refresh_face_tabs)

        safe_after(0, refresh_face_tabs)

    except Exception as e:
        log("🔥 Error during processing:\n" + traceback.format_exc())
        messagebox.showerror("Processing Error", str(e))

    finally:
        safe_after(0, start_btn.config, {"state": NORMAL})
        safe_after(0, stop_btn.config, {"state": DISABLED})
        log("\n🏁 Done.\n")

# ========= Start/Stop/Reset =========
def start_processing():
    global processing_thread, stop_flag
    console_text.configure(state="normal")
    console_text.delete("1.0", END)
    console_text.configure(state="disabled")
    
    # Clear all tabs
    clear_container(reference_faces_frame_inner)
    clear_container(matched_faces_frame_inner)
    clear_container(unmatched_faces_frame_inner)
    clear_container(skipped_faces_frame_inner)
    clear_container(moved_faces_frame_inner)
    
    selected_images.clear() # Clear selection state on new run

    total_videos_var.set(0)
    processed_videos_var.set(0)
    matched_videos_var.set(0)
    not_matched_videos_var.set(0)
    yet_to_process_var.set(0)
    progress_var.set(0)

    with processing_lock:
        stop_flag = False

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
    if processing_thread and processing_thread.is_alive():
        messagebox.showwarning("Busy", "Please wait until processing finishes or click Stop.")
        return

    for var in [reference_folder, video_folder, matched_videos_folder, output_folder]:
        var.set("")
    for var in [total_videos_var, processed_videos_var, matched_videos_var,
                not_matched_videos_var, yet_to_process_var]:
        var.set(0)
    progress_var.set(0)

    # Clear all tabs
    clear_container(reference_faces_frame_inner)
    clear_container(matched_faces_frame_inner)
    clear_container(unmatched_faces_frame_inner)
    clear_container(skipped_faces_frame_inner)
    clear_container(moved_faces_frame_inner)
    
    selected_images.clear() # Clear selection state on reset

    console_text.configure(state="normal")
    console_text.delete("1.0", END)
    console_text.configure(state="disabled")

# ========= GUI =========
root = Tk()
root.title("Face Match Video Search (InsightFace)")
root.geometry("1060x780")

# Create the popup menu for right-click actions
popup_menu = Menu(root, tearoff=0)
popup_menu.add_command(label="Move Selected", command=move_selected_files)

# StringVar definitions
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
tab_stats = Frame(notebook)
tab_ref = Frame(notebook)
tab_matched = Frame(notebook)
tab_unmatched = Frame(notebook)
tab_skipped = Frame(notebook)
tab_moved = Frame(notebook)

# --- CORRECTED: Add tabs sequentially using .add() ---
notebook.add(tab_stats, text="Stats")
notebook.add(tab_ref, text="Reference Faces")
notebook.add(tab_matched, text="Matched Faces")
notebook.add(tab_unmatched, text="Unmatched Faces")
notebook.add(tab_skipped, text="Skipped Faces")
notebook.add(tab_moved, text="Moved Faces")
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

rowpair(tab_stats, 0, "Reference Images Folder:", reference_folder, browse=True,
        cmd=lambda: reference_folder.set(filedialog.askdirectory()))
rowpair(tab_stats, 1, "Videos Folder:", video_folder, browse=True,
        cmd=lambda: video_folder.set(filedialog.askdirectory()))
rowpair(tab_stats, 2, "Matched Videos Folder:", matched_videos_folder, browse=True,
        cmd=lambda: matched_videos_folder.set(filedialog.askdirectory()))
rowpair(tab_stats, 3, "Images Output Base Folder:", output_folder, browse=True,
        cmd=lambda: output_folder.set(filedialog.askdirectory()))

rowpair(tab_stats, 4, "Distance Threshold:", distance_threshold)
rowpair(tab_stats, 5, "Frame Jump:", frame_jump)
rowpair(tab_stats, 6, "Max Frames:", max_frames)

buttons_frame = Frame(tab_stats)
buttons_frame.grid(row=7, column=0, columnspan=3, pady=10, sticky="w")

start_btn = Button(buttons_frame, text="Start Processing", command=start_processing, width=18)
stop_btn = Button(buttons_frame, text="Stop Processing", command=stop_processing, width=16, state=DISABLED)
reset_btn = Button(buttons_frame, text="Reset", command=reset_all, width=10)
refresh_btn = Button(buttons_frame, text="Refresh Thumbnails", command=refresh_face_tabs, width=18)

start_btn.grid(row=0, column=0, padx=4)
stop_btn.grid(row=0, column=1, padx=4)
reset_btn.grid(row=0, column=2, padx=4)
refresh_btn.grid(row=0, column=3, padx=4)

ttk.Progressbar(tab_stats, variable=progress_var, maximum=100).grid(row=8, column=0, columnspan=3, sticky="ew", padx=6, pady=4)

rowpair(tab_stats, 9,  "Total Videos:", total_videos_var, is_entry=False)
rowpair(tab_stats, 10, "Processed:", processed_videos_var, is_entry=False)
rowpair(tab_stats, 11, "Matched:", matched_videos_var, is_entry=False)
rowpair(tab_stats, 12, "Not Matched:", not_matched_videos_var, is_entry=False)
rowpair(tab_stats, 13, "Yet to Process:", yet_to_process_var, is_entry=False)

Label(tab_stats, text="Console Log:").grid(row=14, column=0, sticky="w", padx=6, pady=(8, 2))
console_text = scrolledtext.ScrolledText(tab_stats, height=10, state="disabled", wrap="word")
console_text.grid(row=15, column=0, columnspan=3, sticky="nsew", padx=6, pady=(0, 8))

tab_stats.grid_columnconfigure(1, weight=1)
tab_stats.grid_rowconfigure(15, weight=1)

# ---- Tabs for scrollable thumbnails ----
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

# Create inner frames for all tabs
reference_faces_frame_inner = create_scrollable_tab(tab_ref)
matched_faces_frame_inner = create_scrollable_tab(tab_matched)
unmatched_faces_frame_inner = create_scrollable_tab(tab_unmatched)
skipped_faces_frame_inner = create_scrollable_tab(tab_skipped)
moved_faces_frame_inner = create_scrollable_tab(tab_moved)

# Clear all containers on startup
clear_container(reference_faces_frame_inner)
clear_container(matched_faces_frame_inner)
clear_container(unmatched_faces_frame_inner)
clear_container(skipped_faces_frame_inner)
clear_container(moved_faces_frame_inner)

root.mainloop()