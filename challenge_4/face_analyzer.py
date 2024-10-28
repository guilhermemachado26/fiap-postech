import cv2
from deepface import DeepFace
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import os
from collections import Counter

# Initialize MediaPipe for gesture detection
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def analyze_frame(frame):
    # Analyze frame for face recognition and emotions
    results = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)
    face_data = []
    for face in results:
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        dominant_gender = max(face['gender'], key=face['gender'].get)  # Determine dominant gender
        face_data.append({
            'box': (x, y, w, h),
            'dominant_emotion': face['dominant_emotion'],
            'age': face.get('age'),
            'gender': dominant_gender,  # Only store the dominant gender
        })
    return face_data

def detect_gestures(frame, hand_detector, pose_detector):
    # Detect hand gestures
    gesture_data = []
    hands = hand_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if hands.multi_hand_landmarks:
        for hand_landmarks in hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_data.append("Hand detected")

    # Detect pose for gesture analysis
    pose_data = []
    pose = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        pose_data.append("Pose detected")

    return gesture_data, pose_data

def is_abrupt_movement(prev_frame, curr_frame, threshold=100000):
    # Calculate the difference between frames and detect abrupt motion
    if prev_frame is None:
        return False
    diff = cv2.absdiff(prev_frame, curr_frame)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_score = np.sum(diff_gray)
    return diff_score > threshold

def detect_features(video_path, output_path, summary_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Counters for summarization
    emotion_counter = Counter()
    gesture_counter = Counter()
    total_anomalies = 0

    # Initialize MediaPipe for hand and pose detection and set up for motion detection
    prev_frame = None
    with mp_hands.Hands() as hand_detector, mp_pose.Pose() as pose_detector:
        for _ in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect abrupt movements as anomalies
            if is_abrupt_movement(prev_frame, frame):
                total_anomalies += 1
                cv2.putText(frame, "Anomaly: Abrupt Movement", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Green color for anomaly label

            prev_frame = frame

            # Analyze face and emotion
            face_data = analyze_frame(frame)
            for face in face_data:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

                # Increment emotion count
                dominant_emotion = face['dominant_emotion']
                emotion_counter[dominant_emotion] += 1

                # Annotate frame with dominant emotion and gender
                cv2.putText(frame, f"{dominant_emotion} | Gender: {face['gender']}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Green color text

            # Detect gestures
            gesture_data, pose_data = detect_gestures(frame, hand_detector, pose_detector)
            
            # Increment gesture counts and flag unusual gestures
            for gesture in gesture_data:
                gesture_counter[gesture] += 1
            for pose in pose_data:
                gesture_counter[pose] += 1

            # Check for atypical gestures
            if not gesture_data and not pose_data:
                total_anomalies += 1
                cv2.putText(frame, "Anomaly: Atypical Gesture", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Green color for atypical gesture label

            # Display gesture info on frame
            for idx, gesture in enumerate(gesture_data + pose_data):
                cv2.putText(frame, gesture, (10, 90 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Green color text

            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Write summary to text file
    with open(summary_path, 'w') as summary_file:
        summary_file.write(f"Total Frames Analyzed: {total_frames}\n")
        summary_file.write(f"Total Anomalies Detected: {total_anomalies}\n\n")

        summary_file.write("Emotion Summary:\n")
        for emotion, count in emotion_counter.items():
            summary_file.write(f"{emotion}: {count}\n")

        summary_file.write("\nGesture Summary:\n")
        for gesture, count in gesture_counter.items():
            summary_file.write(f"{gesture}: {count}\n")

    print(f"Summary written to {summary_path}")

# Run the detection function
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video.mp4')
output_video_path = os.path.join(script_dir, 'output_video.mp4')
summary_path = os.path.join(script_dir, 'summary.txt')

detect_features(input_video_path, output_video_path, summary_path)
