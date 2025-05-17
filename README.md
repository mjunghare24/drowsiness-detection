# Drowsiness Detection System 😴🚨

A real-time driver drowsiness detection system using OpenCV, MediaPipe, and EAR (Eye Aspect Ratio). When drowsiness is detected, an audible alarm is triggered to alert the user.

## 📌 Features

- Real-time eye tracking using MediaPipe Face Mesh.
- Eye Aspect Ratio (EAR)-based drowsiness detection.
- Audible alarm when eyes remain closed for too long.
- Uses OpenCV for video processing.
- Minimal dependencies and easy to run locally.

## 🔧 Requirements

Install the following dependencies using pip:

```bash
pip install opencv-python mediapipe pygame numpy
📁 Project Structure
bash
Copy
Edit
drowsiness_detection_project/
│
├── drowsiness_detection.py     # Main script
├── alarm.mp3.wav               # Alarm sound file
└── README.md                   # This file
🚀 How to Run
Place your alarm.mp3.wav in the same folder as the script.

Run the detection script:

bash
Copy
Edit
python drowsiness_detection.py
The webcam will open and begin monitoring. If drowsiness is detected (based on eye closure duration), the alarm will sound.

⚙️ How It Works
Calculates the Eye Aspect Ratio (EAR) from 6 facial landmarks per eye.

If EAR drops below a set threshold for a specified duration, it assumes drowsiness.

Plays an alarm sound to alert the user.

🧠 Algorithms Used
Eye Aspect Ratio (EAR): A robust indicator of whether eyes are open or closed.

MediaPipe Face Mesh: For detecting facial landmarks in real time.

📢 Alarm Source
You can use any .mp3 or .wav file. Just replace alarm.mp3.wav with your preferred sound file (keeping the filename or updating the code).
