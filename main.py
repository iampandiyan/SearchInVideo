import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import face_recognition
import os

# --- Main Application Class ---
class FaceRecognitionApp:
    """
    A simple desktop application for finding a specific person in a video.
    The app uses Tkinter for the GUI, OpenCV for video processing,
    and the face_recognition library for face detection and identification.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Local Face Recognition App")
        self.root.geometry("800x600")

        # --- Variables to store data ---
        self.known_face_encodings = []
        self.known_face_names = []
        self.video_path = None
        self.known_image_path = None
        
        self.cap = None  # Video capture object

        # --- Setup the GUI layout ---
        self.setup_ui()

    def setup_ui(self):
        """
        Create the main user interface elements.
        """
        # Main frames
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        tk.Button(control_frame, text="1. Select Known Person's Image", command=self.load_known_image).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(control_frame, text="2. Select Video File", command=self.select_video).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(control_frame, text="3. Start Processing", command=self.start_processing).pack(side=tk.LEFT, padx=5, pady=5)
        
        self.status_label = tk.Label(control_frame, text="Status: Ready", anchor="w")
        self.status_label.pack(side=tk.RIGHT, padx=5, pady=5)

    def load_known_image(self):
        """
        Prompts the user to select an image of the person to recognize.
        The function then encodes the face in the image for comparison.
        """
        self.known_image_path = filedialog.askopenfilename(
            title="Select Image of Known Person",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not self.known_image_path:
            return

        try:
            # Load the image and find face encodings
            image = face_recognition.load_image_file(self.known_image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                messagebox.showerror("Error", "No face found in the selected image. Please try another image.")
                self.known_image_path = None
                return
            
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(os.path.basename(self.known_image_path).split('.')[0])
            self.status_label.config(text=f"Status: Loaded '{self.known_face_names[0]}'. Ready to process video.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not process the image: {e}")
            self.known_image_path = None

    def select_video(self):
        """
        Prompts the user to select a video file to process.
        """
        self.video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.mov *.avi")]
        )
        if self.video_path:
            self.status_label.config(text="Status: Video selected. Press 'Start Processing'.")

    def start_processing(self):
        """
        Initializes video processing if both an image and a video are selected.
        """
        if not self.known_image_path:
            messagebox.showwarning("Warning", "Please select a known person's image first.")
            return

        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first.")
            return

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open the video file.")
            self.video_path = None
            return
        
        self.status_label.config(text="Status: Processing video...")
        self.process_video_frame()

    def process_video_frame(self):
        """
        Reads one frame from the video, processes it, and displays it.
        This function is called repeatedly to process the entire video.
        """
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize and convert frame for faster face recognition
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all faces and their encodings in the current frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                # Loop through each face found
                for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                    # Compare the current face with the known person's face
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]

                    # Scale back up the face locations to the original frame size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw a box and label around the face
                    if name == "Unknown":
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, "Unknown", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    else:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                # Convert the processed frame to a format Tkinter can display
                img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.canvas.image = img_tk
                self.root.after(10, self.process_video_frame)  # Call the next frame after 10ms
            else:
                self.cap.release()
                self.status_label.config(text="Status: Finished processing video.")
                messagebox.showinfo("Done", "Video processing is complete.")
        
# --- Main entry point ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

