"""
Face Detection and Recognition System
-----------------------------------
A real-time face detection and recognition system using OpenCV and face_recognition.
Features:
- Loads and encodes known faces from a directory
- Detects faces using Haar cascades
- Recognizes faces using face_recognition library
- Works with both images and live webcam feed
- Configurable detection parameters
"""

import cv2
import face_recognition
import os
import numpy as np
from typing import Tuple, List, Optional

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir: str = "known_faces", detection_tolerance: float = 0.6):
        """
        Initialize the face recognition system.
        
        Args:
            known_faces_dir: Directory containing known face images
            detection_tolerance: Tolerance for face recognition (0-1)
        """
        self.known_faces_dir = known_faces_dir
        self.detection_tolerance = detection_tolerance
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.known_encodings, self.known_names = self._load_known_faces()

    def _load_known_faces(self) -> Tuple[List[np.ndarray], List[str]]:
        """Load and encode known faces from directory."""
        known_encodings = []
        known_names = []

        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"[INFO] Created directory: {self.known_faces_dir}")
            return known_encodings, known_names

        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                try:
                    path = os.path.join(self.known_faces_dir, filename)
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)

                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        name = os.path.splitext(filename)[0]
                        known_names.append(name)
                        print(f"[INFO] Loaded {name}")
                    else:
                        print(f"[WARNING] No face found in {filename}")
                except Exception as e:
                    print(f"[ERROR] Failed to process {filename}: {str(e)}")

        return known_encodings, known_names

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Process a single frame and return recognized faces."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        names = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(
                self.known_encodings, 
                face_encoding, 
                tolerance=self.detection_tolerance
            )
            name = "Unknown"

            if len(matches) > 0:
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]

            names.append(name)
            self._draw_face_box(frame, (left, top, right, bottom), name)

        return frame, names

    def _draw_face_box(self, frame: np.ndarray, coords: Tuple[int, int, int, int], name: str):
        """Draw bounding box and name for a detected face."""
        left, top, right, bottom = coords
        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw label background
        cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)
        # Draw name
        cv2.putText(frame, name, (left + 6, top - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

    def run_webcam(self):
        """Run face recognition on webcam feed."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam")
            return

        print("[INFO] Press 'q' to quit the webcam window")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break

            frame, _ = self.process_frame(frame)
            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_image(self, image_path: str) -> Optional[np.ndarray]:
        """Process a single image file."""
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return None

        try:
            image = cv2.imread(image_path)
            processed_image, _ = self.process_frame(image)
            return processed_image
        except Exception as e:
            print(f"[ERROR] Failed to process image: {str(e)}")
            return None

def main():
    # Initialize system
    face_system = FaceRecognitionSystem(
        known_faces_dir="known_faces",
        detection_tolerance=0.6
    )

    # Check if we have any known faces
    if not face_system.known_names:
        print("[WARNING] No known faces loaded. Add images to the known_faces directory.")
        return

    # Process single image
    test_image = "test_images/group_photo.jpg"
    if os.path.exists(test_image):
        result = face_system.process_image(test_image)
        if result is not None:
            cv2.imshow("Detected Faces", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Run webcam detection
    face_system.run_webcam()

if __name__ == "__main__":
    main()