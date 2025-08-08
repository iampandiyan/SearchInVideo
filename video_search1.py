import face_recognition
import cv2
import os
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading

#Added Search Menu
class VideoSearchApp:
    def __init__(self, root):
        """
        Initializes the main application window and its widgets.
        """
        self.root = root
        self.root.title("Video Face Search")
        self.root.geometry("600x500")
        self.root.resizable(False, False)

        # File paths and settings
        self.known_image_path = ""
        self.video_folder_path = ""
        self.known_face_encoding = None
        self.running_search = False
        self.tolerance_var = tk.StringVar(value="0.5") # Default tolerance value

        # Set up GUI components
        self.create_widgets()

    def create_widgets(self):
        """
        Creates and places all the GUI widgets.
        """
        # Main frame to hold everything
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)

        # Frame for image selection
        image_frame = tk.Frame(main_frame)
        image_frame.pack(fill="x", pady=5)
        
        tk.Label(image_frame, text="Person Image to Search:").pack(side="left", padx=(0, 5))
        self.image_path_label = tk.Label(image_frame, text="No image selected", relief="sunken", anchor="w")
        self.image_path_label.pack(side="left", fill="x", expand=True)
        self.image_button = tk.Button(image_frame, text="Select Image", command=self.select_image)
        self.image_button.pack(side="left", padx=(5, 0))

        # Frame for folder selection
        folder_frame = tk.Frame(main_frame)
        folder_frame.pack(fill="x", pady=5)
        
        tk.Label(folder_frame, text="Folder with Videos:").pack(side="left", padx=(0, 5))
        self.folder_path_label = tk.Label(folder_frame, text="No folder selected", relief="sunken", anchor="w")
        self.folder_path_label.pack(side="left", fill="x", expand=True)
        self.folder_button = tk.Button(folder_frame, text="Select Folder", command=self.select_folder)
        self.folder_button.pack(side="left", padx=(5, 0))
        
        # Frame for tolerance setting
        tolerance_frame = tk.Frame(main_frame)
        tolerance_frame.pack(fill="x", pady=5)

        tk.Label(tolerance_frame, text="Matching Tolerance (0.0-1.0):").pack(side="left", padx=(0, 5))
        self.tolerance_entry = tk.Entry(tolerance_frame, textvariable=self.tolerance_var, width=5)
        self.tolerance_entry.pack(side="left")
        tk.Label(tolerance_frame, text="(Lower value = stricter match)").pack(side="left", padx=(5, 0))


        # Search button
        self.search_button = tk.Button(main_frame, text="Search", command=self.start_search_thread)
        self.search_button.pack(fill="x", pady=10)

        # Output text area
        tk.Label(main_frame, text="Search Results:").pack(anchor="w")
        self.output_text = scrolledtext.ScrolledText(main_frame, wrap="word", height=15)
        self.output_text.pack(fill="both", expand=True)
        self.output_text.config(state="disabled") # Make the text box read-only

    def log_message(self, message):
        """
        Displays a message in the GUI's text box.
        """
        self.output_text.config(state="normal")
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END) # Scroll to the bottom
        self.output_text.config(state="disabled")

    def select_image(self):
        """
        Opens a file dialog to select the person's image.
        """
        file_path = filedialog.askopenfilename(
            title="Select Person's Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.known_image_path = file_path
            self.image_path_label.config(text=os.path.basename(file_path))
            self.known_face_encoding = self.get_face_encoding(file_path)

    def select_folder(self):
        """
        Opens a folder dialog to select the videos folder.
        """
        folder_path = filedialog.askdirectory(title="Select Videos Folder")
        if folder_path:
            self.video_folder_path = folder_path
            self.folder_path_label.config(text=os.path.basename(folder_path))

    def get_face_encoding(self, image_path):
        """
        Loads an image and gets its face encoding.
        """
        try:
            known_image = face_recognition.load_image_file(image_path)
            known_face_encodings = face_recognition.face_encodings(known_image)
            if not known_face_encodings:
                messagebox.showerror("Error", "No face found in the provided image.")
                return None
            return known_face_encodings[0]
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image file: {e}")
            return None

    def find_person_in_videos(self):
        """
        The core logic to search for a person in a folder of videos.
        This function runs in a separate thread to prevent the GUI from freezing.
        """
        self.log_message("Starting video search...")
        self.search_button.config(state="disabled")
        self.image_button.config(state="disabled")
        self.folder_button.config(state="disabled")
        self.running_search = True

        try:
            tolerance = float(self.tolerance_var.get())
        except ValueError:
            self.log_message("Invalid tolerance value. Please enter a number between 0.0 and 1.0.")
            self.search_button.config(state="normal")
            self.image_button.config(state="normal")
            self.folder_button.config(state="normal")
            self.running_search = False
            return
        
        found_videos = []
        video_files = [f for f in os.listdir(self.video_folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        if not video_files:
            self.log_message(f"No video files found in the specified folder.")
            self.search_button.config(state="normal")
            self.image_button.config(state="normal")
            self.folder_button.config(state="normal")
            self.running_search = False
            return
        
        for video_file in video_files:
            if not self.running_search:
                break
                
            video_path = os.path.join(self.video_folder_path, video_file)
            self.log_message(f"Checking video: {video_file}...")
            
            video_capture = cv2.VideoCapture(video_path)
            person_found_in_video = False
            
            frame_skip = 1 
            frame_count = 0

            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                for face_encoding in face_encodings:
                    # Use the user-defined tolerance here
                    matches = face_recognition.compare_faces([self.known_face_encoding], face_encoding, tolerance=tolerance)
                    if True in matches:
                        self.log_message(f"Person FOUND in {video_file}!")
                        found_videos.append(video_file)
                        person_found_in_video = True
                        break
                
                if person_found_in_video:
                    break

            video_capture.release()
            if not person_found_in_video:
                self.log_message(f"Person NOT FOUND in {video_file}.")
        
        self.log_message("\nSearch complete!")
        if found_videos:
            self.log_message("The person was found in the following videos:")
            for video in found_videos:
                self.log_message(f"- {video}")
        else:
            self.log_message("The person was not found in any of the videos.")
        
        self.search_button.config(state="normal")
        self.image_button.config(state="normal")
        self.folder_button.config(state="normal")
        self.running_search = False

    def start_search_thread(self):
        """
        Starts the search process in a separate thread to avoid freezing the GUI.
        """
        if not self.known_image_path or not self.video_folder_path:
            messagebox.showerror("Error", "Please select both a person's image and a video folder.")
            return

        if self.known_face_encoding is None:
            messagebox.showerror("Error", "No face was detected in the selected image. Please try another image.")
            return

        # Clear previous results
        self.output_text.config(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state="disabled")

        # Start the search function in a new thread
        search_thread = threading.Thread(target=self.find_person_in_videos)
        search_thread.start()

if __name__ == "__main__":
    # Create the main window and start the application
    root = tk.Tk()
    app = VideoSearchApp(root)
    root.mainloop()
