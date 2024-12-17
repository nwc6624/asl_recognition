import sys
import cv2
import pickle
import numpy as np
import pyttsx3
import mediapipe as mp
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer

# Load the model safely
def load_model(model_file):
    try:
        with open(model_file, "rb") as f:
            loaded_data = pickle.load(f)
        # Handle model structure
        if isinstance(loaded_data, dict):
            model = loaded_data.get("classifier", None)
            if model is None:
                raise ValueError("Model not found in dictionary.")
        else:
            model = loaded_data
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()

# Extract hand landmarks as features
def extract_features(hand_landmarks):
    features = []
    for landmark in hand_landmarks.landmark:
        features.append(landmark.x)
        features.append(landmark.y)
    return np.array(features)

# Main Application Window
class ASLRecognitionApp(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.setWindowTitle("ASL Recognition")
        self.setGeometry(100, 100, 800, 600)

        # Layout
        self.layout = QVBoxLayout()

        # QLabel to show video feed
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(800, 500)
        self.layout.addWidget(self.video_label)

        # QLabel to show recognized text
        self.prediction_label = QLabel("Prediction: ", self)
        self.prediction_label.setStyleSheet("font-size: 20px;")
        self.layout.addWidget(self.prediction_label)

        # Speak Button
        self.speak_button = QPushButton("Speak", self)
        self.speak_button.clicked.connect(self.speak_word)
        self.layout.addWidget(self.speak_button)

        # Main widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

        # Video Capture and Timer
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot access the webcam.")
            sys.exit()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        # State Variables
        self.current_word = ""
        self.model = model

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Flip the frame for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                features = extract_features(hand_landmarks)
                if len(features) == 42:  # 21 landmarks * 2 (x, y)
                    try:
                        prediction = self.model.predict([features])[0]
                        self.current_word += prediction if prediction.isalnum() else ""
                    except Exception as e:
                        print(f"Prediction error: {e}")
                    # Draw landmarks
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Update prediction label
        self.prediction_label.setText(f"Prediction: {self.current_word}")

        # Convert frame to QImage and display it
        h, w, ch = frame.shape
        qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def speak_word(self):
        if self.current_word:
            tts_engine.say(self.current_word)
            tts_engine.runAndWait()
            self.current_word = ""
            self.prediction_label.setText("Prediction: ")

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

# Main function
if __name__ == "__main__":
    # Load the model
    model_file = "model.p"
    model = load_model(model_file)

    # Start the PySide6 application
    app = QApplication(sys.argv)
    window = ASLRecognitionApp(model)
    window.show()
    sys.exit(app.exec())
