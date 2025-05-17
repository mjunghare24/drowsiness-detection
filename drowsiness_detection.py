import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
import pygame

# Initialize pygame mixer for alarm
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3.wav")

def get_mediapipe_app(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
):
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

def distance(p1, p2):
    return sum([(a - b) ** 2 for a, b in zip(p1, p2)]) ** 0.5

def get_ear(landmarks, refer_idxs, w, h):
    try:
        coords = [denormalize_coordinates(landmarks[i].x, landmarks[i].y, w, h) for i in refer_idxs]
        ear = (distance(coords[1], coords[5]) + distance(coords[2], coords[4])) / (2.0 * distance(coords[0], coords[3]))
    except:
        return 0.0, None
    return ear, coords

def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, w, h):
    left_ear, left_coords = get_ear(landmarks, left_eye_idxs, w, h)
    right_ear, right_coords = get_ear(landmarks, right_eye_idxs, w, h)
    return (left_ear + right_ear) / 2.0, (left_coords, right_coords)

def plot_eye_landmarks(frame, left_coords, right_coords, color):
    for eye_coords in [left_coords, right_coords]:
        if eye_coords:
            for coord in eye_coords:
                cv2.circle(frame, coord, 2, color, -1)
    return frame

def plot_text(img, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    return cv2.putText(img, text, origin, font, fntScale, color, thickness)

class VideoFrameHandler:
    def __init__(self):
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }
        self.RED = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.facemesh_model = get_mediapipe_app()
        self.state = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,
            "COLOR": self.GREEN,
            "play_alarm": False,
        }
        self.EAR_txt_pos = (10, 30)

    def process(self, frame: np.array, thresholds: dict):
        frame.flags.writeable = False
        h, w = frame.shape[:2]
        DROWSY_TIME_txt_pos = (10, int(h // 2 * 1.7))
        ALM_txt_pos = (10, int(h // 2 * 1.85))
        results = self.facemesh_model.process(frame)
        frame.flags.writeable = True

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ear, coords = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], w, h)
            frame = plot_eye_landmarks(frame, coords[0], coords[1], self.state["COLOR"])

            if ear < thresholds["EAR_THRESH"]:
                now = time.perf_counter()
                self.state["DROWSY_TIME"] += now - self.state["start_time"]
                self.state["start_time"] = now
                self.state["COLOR"] = self.RED

                if self.state["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state["play_alarm"] = True
                    plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, self.state["COLOR"])
            else:
                self.state["start_time"] = time.perf_counter()
                self.state["DROWSY_TIME"] = 0.0
                self.state["COLOR"] = self.GREEN
                self.state["play_alarm"] = False

            plot_text(frame, f"EAR: {round(ear, 2)}", self.EAR_txt_pos, self.state["COLOR"])
            plot_text(frame, f"DROWSY: {round(self.state['DROWSY_TIME'], 3)} Secs", DROWSY_TIME_txt_pos, self.state["COLOR"])
        else:
            self.state["start_time"] = time.perf_counter()
            self.state["DROWSY_TIME"] = 0.0
            self.state["COLOR"] = self.GREEN
            self.state["play_alarm"] = False

        return cv2.flip(frame, 1), self.state["play_alarm"]

def main():
    cap = cv2.VideoCapture(0)
    frame_handler = VideoFrameHandler()
    thresholds = {"EAR_THRESH": 0.3, "WAIT_TIME": 2.0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_frame, play_alarm = frame_handler.process(frame, thresholds)

        if play_alarm and not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()
        elif not play_alarm and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

        cv2.imshow("Drowsiness Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
