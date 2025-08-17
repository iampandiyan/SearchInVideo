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
ref_embeddings = []
selected_images = set()

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
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img_eq, -1, kernel)

# ========= Reference Handling =========
def load_reference_embeddings(folder_path, ref_output):
    """Load all face embeddings and save cropped faces."""
    embeddings = []
    os.makedirs(ref_output, exist_ok=True)
    for source in [folder_path, os.path.join(folder_path, "manually_selected_faces")]:
        if not os.path.exists(source):
            continue
        for filename in os.listdir(source):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(source, filename)
                img = cv2.imread(img_path)
                if img is None: continue
                faces = app.get(img)
                if not faces: continue
                face = faces[0]
                x1, y1, x2, y2 = map(int, face.bbox)
                
                # --- THIS LINE IS NOW FIXED ---
                # It correctly uses img.shape(see the generated image above) for width and img.shape for height.
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape(see the generated image above), x2), min(img.shape, y2)
                
                cropped_face = img[y1:y2, x1:x2]
                if cropped_face.size > 0:
                    save_path = os.path.join(ref_output, f"ref_{filename}")
                    cv2.imwrite(save_path, cropped_face)
                embeddings.append(face.normed_embedding)
    if not embeddings:
        raise ValueError("❌ No valid reference embeddings found in the selected folder(s).")
    return embeddings

def get_min_distance(emb, ref_list):
    """Return min L2 distance between emb and any embedding in ref_list."""
    if not ref_list: return float("inf")
    refs = np.stack(ref_list, axis=0)
    diffs = refs - emb
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    return float(np.min(dists))

# ========= Thumbnail & File Operations =========
def get_video_base_name(image_path):
    """Extracts video base name from image filename like 'my_video_f123_...jpg'"""
    filename = os.path.basename(image_path)
    parts = filename.split('_f')
    return parts[0] if len(parts) > 1 else None

def move_to_source():
    """Moves selected videos from Matched back to Source folder."""
    if not selected_images:
        messagebox.showinfo("Info", "No images selected.")
        return

    videos_to_process = {get_video_base_name(p) for p in selected_images if get_video_base_name(p)}
    log(f"Move-to-Source operation started for {len(videos_to_process)} video(s).")

    src_video_dir = matched_videos_folder.get()
    dest_video_dir = video_folder.get()
    moved_img_dir = os.path.join(output_folder.get(), "moved_faces")

    if not all([src_video_dir, dest_video_dir, moved_img_dir]):
        messagebox.showerror("Error", "One or more source/destination folders are not set.")
        return

    for base_name in videos_to_process:
        original_video_path = None
        for f in os.listdir(src_video_dir):
            if os.path.splitext(f)[0] == base_name:
                original_video_path = os.path.join(src_video_dir, f)
                break
        if original_video_path and os.path.exists(original_video_path):
            try:
                shutil.move(original_video_path, dest_video_dir)
                log(f"📦 Moved video back to source: {os.path.basename(original_video_path)} -> {dest_video_dir}")
            except Exception as e:
                log(f"⚠️ Error moving video {os.path.basename(original_video_path)}: {e}")
        else:
            log(f"ℹ️ Video for '{base_name}' not found in matched folder.")

    for img_path in selected_images:
        try:
            shutil.move(img_path, moved_img_dir)
        except Exception as e:
            log(f"⚠️ Error moving image {os.path.basename(img_path)}: {e}")

    log("Move-to-Source operation complete.")
    selected_images.clear()
    refresh_face_tabs()

def move_selected_files():
    """Moves selected videos and all their associated images."""
    if not selected_images:
        messagebox.showinfo("Info", "No images selected.")
        return

    videos_to_process = {get_video_base_name(p) for p in selected_images if get_video_base_name(p)}
    log(f"Move operation started for {len(videos_to_process)} video(s).")

    src_video_dir = video_folder.get()
    dest_video_dir = matched_videos_folder.get()
    unmatched_img_dir = os.path.join(output_folder.get(), "not_matched_faces")
    skipped_img_dir = os.path.join(output_folder.get(), "skipped_faces")
    moved_img_dir = os.path.join(output_folder.get(), "moved_faces")

    if not all([src_video_dir, dest_video_dir, output_folder.get()]):
        messagebox.showerror("Error", "One or more source/destination folders are not set.")
        return

    for base_name in videos_to_process:
        original_video_path = None
        for f in os.listdir(src_video_dir):
            if os.path.splitext(f)[0] == base_name:
                original_video_path = os.path.join(src_video_dir, f)
                break
        if original_video_path and os.path.exists(original_video_path):
            try:
                shutil.move(original_video_path, dest_video_dir)
                log(f"📦 Moved video: {os.path.basename(original_video_path)} -> {dest_video_dir}")
            except Exception as e:
                log(f"⚠️ Error moving video {os.path.basename(original_video_path)}: {e}")
        else:
            log(f"ℹ️ Video for '{base_name}' not found in source folder.")

        for folder in [unmatched_img_dir, skipped_img_dir]:
            if not os.path.exists(folder): continue
            for img_file in os.listdir(folder):
                if img_file.startswith(base_name + '_'):
                    try:
                        shutil.move(os.path.join(folder, img_file), moved_img_dir)
                    except Exception as e:
                        log(f"⚠️ Error moving image {img_file}: {e}")

    log("Move operation complete.")
    selected_images.clear()
    refresh_face_tabs()

def toggle_selection(widget, img_path):
    """Add/remove image from selection and update widget appearance."""
    if img_path in selected_images:
        selected_images.remove(img_path)
        widget.config(relief="flat", borderwidth=1, bg="#F0F2F5") 
    else:
        selected_images.add(img_path)
        widget.config(relief="solid", borderwidth=2, bg="#A8C7FA")

def show_popup_menu(event, widget, menu_type):
    """Show the correct right-click menu."""
    img_path = widget.img_path
    toggle_selection(widget, img_path)
    menu = popup_menu if menu_type == 'unmatched' else popup_menu_matched
    menu.post(event.x_root, event.y_root)

def clear_container(container):
    for w in container.winfo_children(): w.destroy()

def open_preview(img_path):
    """Popup window to preview a face image."""
    try:
        win = Toplevel(root)
        win.title(os.path.basename(img_path))
        win.configure(bg="#F0F2F5")
        img = Image.open(img_path).convert("RGB")
        max_side = 640
        ratio = min(max_side / img.width, max_side / img.height, 1.0)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        tkimg = ImageTk.PhotoImage(img)
        lbl = Label(win, image=tkimg, bg="#F0F2F5")
        lbl.image = tkimg
        lbl.pack(padx=10, pady=10)
    except Exception as e:
        messagebox.showerror("Preview error", str(e))

def on_thumbnail_click(event, img_path, widget):
    """Handles left-clicks for preview or multi-selection."""
    if event.state & 4 or event.state & 1: # Ctrl or Shift
        toggle_selection(widget, img_path)
    else:
        open_preview(img_path)

def load_faces_to_tab(folder, container, tab_name=None):
    """Fill a tab with thumbnails and set up click/right-click bindings."""
    clear_container(container)
    if not os.path.exists(folder): return
    row, col = 0, 0
    for file in sorted(os.listdir(folder)):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder, file)
            try:
                img = Image.open(img_path).convert("RGB").resize((110, 110))
                img_tk = ImageTk.PhotoImage(img)
                btn = Button(container, image=img_tk, relief="flat", borderwidth=1, bg="#F0F2F5")
                btn.image = img_tk
                btn.img_path = img_path

                if tab_name in ["unmatched", "skipped", "matched"]:
                    btn.bind("<Button-1>", lambda e, p=img_path, w=btn: on_thumbnail_click(e, p, w))
                    menu_type = 'matched' if tab_name == "matched" else 'unmatched'
                    btn.bind("<Button-3>", lambda e, w=btn, mt=menu_type: show_popup_menu(e, w, mt))
                else:
                    btn.config(command=lambda p=img_path: open_preview(p))

                btn.grid(row=row, column=col, padx=4, pady=4)
                if img_path in selected_images:
                    btn.config(relief="solid", borderwidth=2, bg="#A8C7FA")
                
                col += 1
                if col >= 6: col, row = 0, row + 1
            except Exception:
                continue
    container.update_idletasks()

def refresh_face_tabs():
    """Refresh all face tabs from the current output folders."""
    paths = {
        "ref": os.path.join(output_folder.get(), "reference_faces"),
        "matched": os.path.join(output_folder.get(), "matched_faces"),
        "unmatched": os.path.join(output_folder.get(), "not_matched_faces"),
        "skipped": os.path.join(output_folder.get(), "skipped_faces"),
        "moved": os.path.join(output_folder.get(), "moved_faces")
    }
    load_faces_to_tab(paths["ref"], reference_faces_frame_inner)
    load_faces_to_tab(paths["matched"], matched_faces_frame_inner, "matched")
    load_faces_to_tab(paths["unmatched"], unmatched_faces_frame_inner, "unmatched")
    load_faces_to_tab(paths["skipped"], skipped_faces_frame_inner, "skipped")
    load_faces_to_tab(paths["moved"], moved_faces_frame_inner, "moved")

# --- Video Processing, Start, Stop, Reset ---
def process_videos():
    global stop_flag, ref_embeddings
    try:
        with processing_lock: stop_flag = False
        safe_after(0, start_btn.config, {"state": DISABLED})
        safe_after(0, stop_btn.config, {"state": NORMAL})
        for var, name in [(reference_folder, "Reference"), (video_folder, "Videos"), (matched_videos_folder, "Matched"), (output_folder, "Output")]:
            if not var.get() or not os.path.isdir(var.get()): raise ValueError(f"Please select a valid {name} Folder.")
        ref_out_dir = os.path.join(output_folder.get(), "reference_faces")
        log("Loading reference embeddings...")
        ref_embeddings = load_reference_embeddings(reference_folder.get(), ref_out_dir)
        log(f"Loaded {len(ref_embeddings)} reference embeddings.")
        folders_to_create = ["matched_faces", "not_matched_faces", "skipped_faces", "moved_faces"]
        for folder in folders_to_create: os.makedirs(os.path.join(output_folder.get(), folder), exist_ok=True)
        video_files = [f for f in os.listdir(video_folder.get()) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
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
        log(f"Starting processing of {total} video(s)...")
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
                if not ret: break
                frame_count += frame_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                faces = app.get(frame)
                if not faces: continue
                for fi, face in enumerate(faces):
                    x1, y1, x2, y2 = map(int, face.bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape, y2)
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size == 0: continue
                    candidate = cropped
                    if candidate.shape < 50 or candidate.shape(see the generated image above) < 50 or is_blurry(candidate):
                        candidate = preprocess_face(candidate)
                    if candidate.shape < 50 or candidate.shape(see the generated image above) < 50:
                        outp = os.path.join(output_folder.get(), "skipped_faces", f"{os.path.splitext(video_name)}_f{frame_count:06d}_face{fi}.jpg")
                        cv2.imwrite(outp, candidate)
                        continue
                    dist = get_min_distance(face.normed_embedding, ref_embeddings)
                    if dist < dist_thr:
                        outp = os.path.join(output_folder.get(), "matched_faces", f"{os.path.splitext(video_name)[0]}_f{frame_count:06d}_match.jpg")
                        cv2.imwrite(outp, candidate)
                        match_found = True
                        ref_embeddings.append(face.normed_embedding)
                        log(f"✅ Match at frame {frame_count} (dist {dist:.4f})")
                        break
                    else:
                        outp = os.path.join(output_folder.get(), "not_matched_faces", f"{os.path.splitext(video_name)[0]}_f{frame_count:06d}_nomatch.jpg")
                        cv2.imwrite(outp, candidate)
                if match_found: break
            cap.release()
            safe_after(0, processed_videos_var.set, processed_videos_var.get() + 1)
            if match_found:
                safe_after(0, matched_videos_var.set, matched_videos_var.get() + 1)
                try:
                    shutil.move(video_path, os.path.join(matched_videos_folder.get(), video_name))
                    log(f"📦 Moved matched video to {matched_videos_folder.get()}")
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
    except Exception as e:
        log("🔥 Error during processing:\n" + traceback.format_exc())
        messagebox.showerror("Processing Error", str(e))
    finally:
        safe_after(0, start_btn.config, {"state": NORMAL})
        safe_after(0, stop_btn.config, {"state": DISABLED})
        log("\n🏁 Done.\n")
        safe_after(0, update_button_states)

def start_processing():
    console_text.configure(state="normal")
    console_text.delete("1.0", END)
    console_text.configure(state="disabled")
    for frame in [reference_faces_frame_inner, matched_faces_frame_inner, unmatched_faces_frame_inner, skipped_faces_frame_inner, moved_faces_frame_inner]:
        clear_container(frame)
    selected_images.clear()
    threading.Thread(target=process_videos, daemon=True).start()

def stop_processing():
    global stop_flag
    with processing_lock: stop_flag = True
    log("🛑 Stop requested. Finishing current step...")

def reset_all():
    if processing_thread and processing_thread.is_alive():
        messagebox.showwarning("Busy", "Please wait until processing finishes or click Stop.")
        return
    for var in [reference_folder, video_folder, matched_videos_folder, output_folder]: var.set("")
    for var in [total_videos_var, processed_videos_var, matched_videos_var, not_matched_videos_var, yet_to_process_var]: var.set(0)
    progress_var.set(0)
    for frame in [reference_faces_frame_inner, matched_faces_frame_inner, unmatched_faces_frame_inner, skipped_faces_frame_inner, moved_faces_frame_inner]: clear_container(frame)
    selected_images.clear()
    console_text.configure(state="normal")
    console_text.delete("1.0", END)
    console_text.configure(state="disabled")

# ========= GUI SETUP =========
root = Tk()
root.title("Face Match Video Search (InsightFace)")
root.geometry("1060x780")

# --- THEME AND STYLE SETUP ---
BG_COLOR = "#F0F2F5"
TEXT_COLOR = "#333333"
PRIMARY_COLOR = "#4A90E2"
PRIMARY_LIGHT = "#D1E2F9"
ACCENT_COLOR = "#A8C7FA"
CONSOLE_BG = "#1E1E1E"
CONSOLE_FG = "#D4D4D4"
BUTTON_FG = "#FFFFFF"
BUTTON_HOVER = "#357ABD"

root.configure(bg=BG_COLOR)

style = ttk.Style()
style.theme_use('clam')

style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
style.configure("TNotebook.Tab", background=PRIMARY_LIGHT, foreground=TEXT_COLOR, font=('Helvetica', 10, 'bold'), padding=[12, 8], borderwidth=1)
style.map("TNotebook.Tab", background=[("selected", PRIMARY_COLOR), ("!selected", PRIMARY_LIGHT)], foreground=[("selected", BUTTON_FG), ("!selected", TEXT_COLOR)])
style.configure("TProgressbar", troughcolor=PRIMARY_LIGHT, background=PRIMARY_COLOR, thickness=20)

popup_menu = Menu(root, tearoff=0, bg=BG_COLOR, fg=TEXT_COLOR, activebackground=PRIMARY_COLOR, activeforeground=BUTTON_FG)
popup_menu.add_command(label="Move Selected", command=move_selected_files)
popup_menu_matched = Menu(root, tearoff=0, bg=BG_COLOR, fg=TEXT_COLOR, activebackground=PRIMARY_COLOR, activeforeground=BUTTON_FG)
popup_menu_matched.add_command(label="Move to Source", command=move_to_source)

reference_folder = StringVar()
video_folder = StringVar()
matched_videos_folder = StringVar()
output_folder = StringVar()
distance_threshold = StringVar(value="0.8")
frame_jump = StringVar(value="10")
max_frames = StringVar(value="5000")
progress_var = DoubleVar(value=0.0)
total_videos_var, processed_videos_var, matched_videos_var, not_matched_videos_var, yet_to_process_var = (IntVar(value=0) for _ in range(5))

def update_button_states(*args):
    all_filled = all(var.get() for var in [reference_folder, video_folder, matched_videos_folder, output_folder])
    new_state = "normal" if all_filled else "disabled"
    start_btn.config(state=new_state)
    refresh_btn.config(state=new_state)

for var in [reference_folder, video_folder, matched_videos_folder, output_folder]:
    var.trace_add("write", update_button_states)

notebook = ttk.Notebook(root, style="TNotebook")
tab_names = ["Stats", "Reference Faces", "Matched Faces", "Unmatched Faces", "Skipped Faces", "Moved Faces"]
tabs = {name: Frame(notebook, bg=BG_COLOR) for name in tab_names}
for name, frame in tabs.items(): notebook.add(frame, text=name)
notebook.pack(expand=True, fill="both", padx=10, pady=(5, 10))

def rowpair(parent, r, label, var, browse=False, cmd=None):
    Label(parent, text=label, bg=BG_COLOR, fg=TEXT_COLOR, font=('Helvetica', 10)).grid(row=r, column=0, sticky="w", padx=6, pady=4)
    entry = Entry(parent, textvariable=var, font=('Helvetica', 10), relief="solid", borderwidth=1, fg=TEXT_COLOR, bg="#FFFFFF")
    entry.grid(row=r, column=1, sticky="ew", padx=6)
    if browse:
        browse_btn = Button(parent, text="Browse", command=cmd, bg=PRIMARY_LIGHT, fg=TEXT_COLOR, activebackground=BUTTON_HOVER, activeforeground=BUTTON_FG, font=('Helvetica', 9, 'bold'), relief="raised", borderwidth=2)
        browse_btn.grid(row=r, column=2, padx=6)

tab_stats = tabs["Stats"]
rowpair(tab_stats, 0, "Reference Images Folder:", reference_folder, True, lambda: reference_folder.set(filedialog.askdirectory()))
rowpair(tab_stats, 1, "Videos Folder:", video_folder, True, lambda: video_folder.set(filedialog.askdirectory()))
rowpair(tab_stats, 2, "Matched Videos Folder:", matched_videos_folder, True, lambda: matched_videos_folder.set(filedialog.askdirectory()))
rowpair(tab_stats, 3, "Images Output Base Folder:", output_folder, True, lambda: output_folder.set(filedialog.askdirectory()))
rowpair(tab_stats, 4, "Distance Threshold:", distance_threshold)
rowpair(tab_stats, 5, "Frame Jump:", frame_jump)
rowpair(tab_stats, 6, "Max Frames:", max_frames)

buttons_frame = Frame(tab_stats, bg=BG_COLOR)
buttons_frame.grid(row=7, column=0, columnspan=3, pady=10, sticky="w")

btn_style_args = {'font': ('Helvetica', 10, 'bold'), 'fg': BUTTON_FG, 'activeforeground': BUTTON_FG, 'relief': 'raised', 'borderwidth': 2}
start_btn = Button(buttons_frame, text="Start Processing", command=start_processing, width=18, bg=PRIMARY_COLOR, activebackground=BUTTON_HOVER, **btn_style_args)
stop_btn = Button(buttons_frame, text="Stop Processing", command=stop_processing, width=16, bg="#E74C3C", activebackground="#C0392B", state="disabled", **btn_style_args)
reset_btn = Button(buttons_frame, text="Reset", command=reset_all, width=10, bg="#F39C12", activebackground="#D35400", **btn_style_args)
refresh_btn = Button(buttons_frame, text="Refresh Thumbnails", command=refresh_face_tabs, width=18, bg=PRIMARY_COLOR, activebackground=BUTTON_HOVER, **btn_style_args)
start_btn.grid(row=0, column=0, padx=4); stop_btn.grid(row=0, column=1, padx=4)
reset_btn.grid(row=0, column=2, padx=4); refresh_btn.grid(row=0, column=3, padx=4)

ttk.Progressbar(tab_stats, variable=progress_var, maximum=100, style="TProgressbar").grid(row=8, column=0, columnspan=3, sticky="ew", padx=6, pady=4)

stats_frame = Frame(tab_stats, bg=BG_COLOR)
stats_frame.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(10, 0))
for i, (label, var) in enumerate([("Total Videos:", total_videos_var), ("Processed:", processed_videos_var), ("Matched:", matched_videos_var), ("Not Matched:", not_matched_videos_var), ("Pending:", yet_to_process_var)]):
    Label(stats_frame, text=label, bg=BG_COLOR, fg=TEXT_COLOR, font=('Helvetica', 10, 'bold')).grid(row=i, column=0, sticky="w", padx=6)
    Label(stats_frame, textvariable=var, bg=BG_COLOR, fg=PRIMARY_COLOR, font=('Helvetica', 10, 'bold')).grid(row=i, column=1, sticky="w", padx=6)

Label(tab_stats, text="Console Log:", bg=BG_COLOR, fg=TEXT_COLOR, font=('Helvetica', 10, 'bold')).grid(row=10, column=0, sticky="w", padx=6, pady=(8, 2))
console_text = scrolledtext.ScrolledText(tab_stats, height=10, state="disabled", wrap="word", bg=CONSOLE_BG, fg=CONSOLE_FG, insertbackground=BUTTON_FG, relief="solid", borderwidth=1)
console_text.grid(row=11, column=0, columnspan=3, sticky="nsew", padx=6, pady=(0, 8))
tab_stats.grid_columnconfigure(1, weight=1); tab_stats.grid_rowconfigure(11, weight=1)

def create_scrollable_tab(tab):
    canvas = Canvas(tab, borderwidth=0, bg=BG_COLOR, highlightthickness=0)
    scrollbar = Scrollbar(tab, orient="vertical", command=canvas.yview)
    scroll_frame = Frame(canvas, bg=BG_COLOR)
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return scroll_frame

reference_faces_frame_inner = create_scrollable_tab(tabs["Reference Faces"])
matched_faces_frame_inner = create_scrollable_tab(tabs["Matched Faces"])
unmatched_faces_frame_inner = create_scrollable_tab(tabs["Unmatched Faces"])
skipped_faces_frame_inner = create_scrollable_tab(tabs["Skipped Faces"])
moved_faces_frame_inner = create_scrollable_tab(tabs["Moved Faces"])

update_button_states()
root.mainloop()
