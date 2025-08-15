import cv2
import numpy as np
import os
import shutil
import threading
from tkinter import *
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis
#Working with the tabs
# ================= InsightFace =================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ================= Globals =================
stop_flag = False
processing_thread = None
processing_lock = threading.Lock()
ref_embeddings = []

# ================= Helper Functions =================
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

def safe_after(ms, fn, *args, **kwargs):
    try:
        root.after(ms, fn, *args, **kwargs)
    except Exception:
        pass

def log(msg):
    print(msg, flush=True)
    if console_text:
        console_text.configure(state="normal")
        console_text.insert(END, msg + "\n")
        console_text.see(END)
        console_text.configure(state="disabled")

# ================= Load Reference Embeddings =================
def load_reference_embeddings(folder_path, ref_output):
    embeddings = []
    os.makedirs(ref_output, exist_ok=True)
    all_sources = [folder_path, os.path.join(folder_path, "manually_selected_faces")]
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
                    continue
                emb = faces[0].normed_embedding
                embeddings.append(emb)
    if not embeddings:
        raise ValueError("❌ No valid reference embeddings found.")
    return embeddings

def get_min_distance(emb, ref_list):
    if not ref_list:
        return float("inf")
    refs = np.stack(ref_list, axis=0)
    diffs = refs - emb
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    return float(np.min(dists))

# ================= GUI Helpers =================
def clear_container(container):
    for w in container.winfo_children():
        w.destroy()

def open_preview(img_path):
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

def create_scrollable_tab(tab):
    canvas = Canvas(tab)
    scrollbar = Scrollbar(tab, orient="vertical", command=canvas.yview)
    scroll_frame = Frame(canvas)
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0,0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return scroll_frame

# ================= Thumbnail Loaders =================
def load_faces_to_tab(folder, container, move_to_matched=False):
    clear_container(container)
    if not os.path.exists(folder):
        return
    row, col = 0, 0
    for file in sorted(os.listdir(folder)):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder, file)
            try:
                img = Image.open(img_path).convert("RGB").resize((110,110))
                img_tk = ImageTk.PhotoImage(img)
                def on_click(p=img_path):
                    if move_to_matched:
                        move_face_to_matched(p)
                btn = Button(container, image=img_tk, command=on_click)
                btn.image = img_tk
                btn.grid(row=row, column=col, padx=4, pady=4)
                col += 1
                if col >= 6:
                    col = 0
                    row +=1
            except Exception:
                continue
    container.update_idletasks()

def refresh_face_tabs():
    load_faces_to_tab(os.path.join(output_folder.get(),"matched_faces"), matched_faces_frame_inner)
    load_faces_to_tab(os.path.join(output_folder.get(),"not_matched_faces"), unmatched_faces_frame_inner, move_to_matched=True)
    load_faces_to_tab(os.path.join(output_folder.get(),"skipped_faces"), skipped_faces_frame_inner, move_to_matched=True)

# ================= Move Face to Matched =================
def move_face_to_matched(face_img_path):
    try:
        matched_folder = os.path.join(output_folder.get(), "matched_faces")
        os.makedirs(matched_folder, exist_ok=True)
        shutil.move(face_img_path, matched_folder)
        # Try moving corresponding video
        video_name = "_".join(os.path.basename(face_img_path).split("_")[:1])
        src_video_path = os.path.join(video_folder.get(), video_name)
        dest_video_path = os.path.join(matched_videos_folder.get(), video_name)
        if os.path.exists(src_video_path):
            shutil.move(src_video_path, dest_video_path)
            log(f"✅ Moved face and video {video_name} to matched folders")
        else:
            messagebox.showinfo("Video Not Found", "Video is not available. It may be already moved.")
        refresh_face_tabs()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# ================= Video Processing =================
def process_videos():
    global stop_flag, ref_embeddings
    stop_flag = False
    try:
        ref_embeddings = load_reference_embeddings(reference_folder.get(), os.path.join(output_folder.get(),"reference_faces"))
        threshold = float(distance_threshold.get())
        frame_step = int(frame_jump.get())
        max_fr = int(max_frames.get())
        vid_files = [f for f in os.listdir(video_folder.get()) if f.lower().endswith((".mp4",".avi",".mov",".mkv"))]
        total_videos_var.set(len(vid_files))
        processed_videos_var.set(0)
        matched_videos_var.set(0)
        not_matched_videos_var.set(0)
        yet_to_process_var.set(len(vid_files))
        progress_var.set(0)

        for idx, video_name in enumerate(sorted(vid_files), start=1):
            if stop_flag:
                log("🛑 Stop requested")
                break
            video_path = os.path.join(video_folder.get(), video_name)
            log(f"Processing: {video_name}")
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            match_found = False
            while cap.isOpened() and frame_count < max_fr:
                if stop_flag:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += frame_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                faces = app.get(frame)
                for fi, face in enumerate(faces):
                    x1, y1, x2, y2 = map(int, face.bbox)
                    x1, y1 = max(0,x1), max(0,y1)
                    x2, y2 = min(frame.shape[1],x2), min(frame.shape[0],y2)
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size==0: continue
                    if cropped.shape[0]<50 or cropped.shape[1]<50 or is_blurry(cropped):
                        cropped = preprocess_face(cropped)
                    dist = get_min_distance(face.normed_embedding, ref_embeddings)
                    if dist < threshold:
                        outp = os.path.join(output_folder.get(),"matched_faces", f"{video_name}_f{frame_count:06d}.jpg")
                        cv2.imwrite(outp, cropped)
                        match_found=True
                        break
                    else:
                        outp = os.path.join(output_folder.get(),"not_matched_faces", f"{video_name}_f{frame_count:06d}.jpg")
                        cv2.imwrite(outp, cropped)
                if match_found:
                    break
            cap.release()
            processed_videos_var.set(processed_videos_var.get()+1)
            yet_to_process_var.set(len(vid_files)-processed_videos_var.get())
            if match_found:
                matched_videos_var.set(matched_videos_var.get()+1)
                try:
                    shutil.move(video_path, os.path.join(matched_videos_folder.get(), video_name))
                except:
                    pass
            else:
                not_matched_videos_var.set(not_matched_videos_var.get()+1)
            progress_var.set((idx/len(vid_files))*100)
            root.update_idletasks()
        log("🏁 Processing Finished")
        refresh_face_tabs()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def start_processing():
    global processing_thread
    processing_thread = threading.Thread(target=process_videos, daemon=True)
    processing_thread.start()

def stop_processing():
    global stop_flag
    stop_flag = True
    log("🛑 Stop requested by user")

# ================= GUI =================
root = Tk()
root.title("Face Match Video Tool")
root.geometry("1100x750")

# Variables
reference_folder = StringVar()
video_folder = StringVar()
matched_videos_folder = StringVar()
output_folder = StringVar()
distance_threshold = StringVar(value="0.8")
frame_jump = StringVar(value="10")
max_frames = StringVar(value="5000")
progress_var = DoubleVar()

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

# ---- Tab1 Controls ----
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
Button(buttons_frame, text="Start Processing", command=start_processing).grid(row=0,column=0,padx=4)
Button(buttons_frame, text="Stop Processing", command=stop_processing).grid(row=0,column=1,padx=4)
progress_bar = ttk.Progressbar(tab1, variable=progress_var, maximum=100)
progress_bar.grid(row=8, column=0, columnspan=3, sticky="ew", padx=6, pady=4)

# Console
Label(tab1, text="Console Log:").grid(row=9, column=0, sticky="w", padx=6, pady=(8,2))
console_text = scrolledtext.ScrolledText(tab1, height=10, state="disabled", wrap="word")
console_text.grid(row=10, column=0, columnspan=3, sticky="nsew", padx=6, pady=(0,8))

tab1.grid_columnconfigure(1, weight=1)
tab1.grid_rowconfigure(10, weight=1)

# ---- Tabs 2-4 Scrollable ----
matched_faces_frame_inner = create_scrollable_tab(tab2)
unmatched_faces_frame_inner = create_scrollable_tab(tab3)
skipped_faces_frame_inner = create_scrollable_tab(tab4)

refresh_face_tabs()

root.mainloop()
