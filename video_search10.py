import sys, os, cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QLabel, QVBoxLayout, QPushButton, QTextEdit, QWidget
from deepface import DeepFace

class FaceSearchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Search in Videos")
        self.resize(600, 400)

        layout = QVBoxLayout()

        self.label_image = QLabel("Select Person Image")
        layout.addWidget(self.label_image)

        btn_select_image = QPushButton("Select Image")
        btn_select_image.clicked.connect(self.select_image)
        layout.addWidget(btn_select_image)

        self.label_folder = QLabel("Select Video Folder")
        layout.addWidget(self.label_folder)

        btn_select_folder = QPushButton("Select Folder")
        btn_select_folder.clicked.connect(self.select_folder)
        layout.addWidget(btn_select_folder)

        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_videos)
        layout.addWidget(self.search_button)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        layout.addWidget(self.result_box)

        self.setLayout(layout)
        self.person_image = None
        self.video_folder = None

    def select_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.png)")
        if file:
            self.person_image = file
            self.label_image.setText(f"Selected Image: {os.path.basename(file)}")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if folder:
            self.video_folder = folder
            self.label_folder.setText(f"Selected Folder: {folder}")

    def search_videos(self):
        if not self.person_image or not self.video_folder:
            self.result_box.append("⚠ Please select both image and folder.")
            return

        os.makedirs("results", exist_ok=True)

        videos = [f for f in os.listdir(self.video_folder) if f.lower().endswith((".mp4", ".avi", ".mov"))]
        self.result_box.clear()
        self.result_box.append("🔍 Starting search...\n")

        for video in videos:
            video_path = os.path.join(self.video_folder, video)
            cap = cv2.VideoCapture(video_path)
            found = False
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Only check every 10th frame for speed
                if frame_count % 10 != 0:
                    continue

                try:
                    result = DeepFace.verify(
                        img1_path=self.person_image,
                        img2_path=frame,
                        model_name="Facenet512",  # more accurate
                        detector_backend="retinaface",  # better for varied angles
                        enforce_detection=True,
                        distance_metric="cosine"
                    )
                    similarity_score = 1 - result['distance']  # Cosine similarity

                    if result["verified"] and similarity_score > 0.65:  # stricter threshold
                        found = True
                        save_path = f"results/{video}_found.jpg"
                        cv2.imwrite(save_path, frame)
                        self.result_box.append(f"✅ {video}: Person Found! (Saved: {save_path})")
                        break
                except Exception as e:
                    self.result_box.append(f"Error in {video}: {str(e)}")
                    break

            cap.release()
            if not found:
                self.result_box.append(f"❌ {video}: Person Not Found.")

        self.result_box.append("\n✅ Search Complete!")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FaceSearchApp()
    win.show()
    sys.exit(app.exec_())
