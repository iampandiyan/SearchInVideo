import sys
import cv2
import numpy as np
import insightface
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QTextEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from insightface.app import FaceAnalysis

class FaceSearchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Search in Video")
        self.setGeometry(200, 200, 800, 600)

        self.ref_img = None
        self.ref_embedding = None
        
        # CPU-only version
        self.model = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=-1, det_size=(640, 640))  # CPU ctx_id=-1

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.image_label = QLabel("No reference image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.load_image_btn = QPushButton("Load Reference Image")
        self.load_image_btn.clicked.connect(self.load_reference_image)
        layout.addWidget(self.load_image_btn)

        self.load_video_btn = QPushButton("Load Video and Search")
        self.load_video_btn.clicked.connect(self.search_in_video)
        layout.addWidget(self.load_video_btn)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        layout.addWidget(self.result_box)

        self.setLayout(layout)

    def load_reference_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Reference Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.ref_img = cv2.imread(file_path)
            faces = self.model.get(self.ref_img)
            if len(faces) == 0:
                self.result_box.append("No face found in the reference image.")
                return
            self.ref_embedding = faces[0].normed_embedding

            # Display the reference image
            rgb_img = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            qimg = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(300, 300, Qt.KeepAspectRatio))
            self.result_box.append("Reference image loaded successfully.")

    def search_in_video(self):
        if self.ref_embedding is None:
            self.result_box.append("Please load a reference image first.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not file_path:
            return

        cap = cv2.VideoCapture(file_path)
        frame_count = 0
        matches_found = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            faces = self.model.get(frame)
            for face in faces:
                sim = np.dot(self.ref_embedding, face.normed_embedding)
                if sim > 0.35:  # similarity threshold
                    matches_found += 1
                    self.result_box.append(f"Match found at frame {frame_count} (Similarity: {sim:.2f})")
                    cv2.imwrite(f"match_frame_{frame_count}.jpg", frame)

        cap.release()
        self.result_box.append(f"Search completed. {matches_found} matches found.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceSearchApp()
    window.show()
    sys.exit(app.exec_())
