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
            'gender': dominant_gender,
        })
    return face_data

def detect_gestures(frame, hand_detector, pose_detector, prev_pose_landmarks, frame_count):
    gesture_data = []
    hands = hand_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Detect hand gestures
    if hands.multi_hand_landmarks:
        for hand_landmarks in hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_data.append("Hand detected")

    # Detect pose for specific gestures (arm up, arm down, walking, dancing)
    pose_data = []
    pose = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if pose.pose_landmarks:
        landmarks = pose.pose_landmarks.landmark

        # Arm up/down detection
        if landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y:
            pose_data.append("Left Arm Up")
        if landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y > landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y:
            pose_data.append("Left Arm Down")
        if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y:
            pose_data.append("Right Arm Up")
        if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y:
            pose_data.append("Right Arm Down")

        # Initialize ankle movement to 0 if no previous landmarks are available
        left_ankle_movement = 0.0
        right_ankle_movement = 0.0

        # Walking detection based on ankle movement
        if prev_pose_landmarks is not None:
            left_ankle_movement = abs(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y - prev_pose_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y)
            right_ankle_movement = abs(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y - prev_pose_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y)
            if left_ankle_movement > 0.02 or right_ankle_movement > 0.02:
                pose_data.append("Walking")

        # Dancing detection based on simultaneous arm and leg movements
        left_arm_movement = abs(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y - prev_pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y) if prev_pose_landmarks else 0
        right_arm_movement = abs(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y - prev_pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y) if prev_pose_landmarks else 0
        left_leg_movement = left_ankle_movement
        right_leg_movement = right_ankle_movement

        # Thresholds for detecting dancing (simultaneous arm and leg movements)
        if (left_arm_movement > 0.05 and right_arm_movement > 0.05) and (left_leg_movement > 0.02 and right_leg_movement > 0.02):
            pose_data.append("Dancing")

        # Update previous landmarks for the next frame
        prev_pose_landmarks = landmarks

    return gesture_data, pose_data, prev_pose_landmarks

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
    arm_movement_counter = Counter()
    activity_counter = Counter()
    total_anomalies = 0
    prev_frame = None
    prev_pose_landmarks = None

    with mp_hands.Hands() as hand_detector, mp_pose.Pose() as pose_detector:
        for frame_count in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            # Detect abrupt movements as anomalies
            if is_abrupt_movement(prev_frame, frame):
                total_anomalies += 1
                cv2.putText(frame, "Anomaly: Abrupt Movement", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            prev_frame = frame

            # Analyze face and emotion
            face_data = analyze_frame(frame)
            for face in face_data:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y, x + w, y + h), (0, 255, 0), 2)
                dominant_emotion = face['dominant_emotion']
                emotion_counter[dominant_emotion] += 1
                cv2.putText(frame, f"{dominant_emotion} | Gender: {face['gender']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Detect gestures and categorize them
            gesture_data, pose_data, prev_pose_landmarks = detect_gestures(frame, hand_detector, pose_detector, prev_pose_landmarks, frame_count)

            # Increment gesture counts
            for gesture in gesture_data:
                activity_counter[gesture] += 1
            for pose in pose_data:
                if "Arm" in pose:
                    arm_movement_counter[pose] += 1
                elif pose in ["Walking", "Dancing"]:
                    activity_counter[pose] += 1

            # Display detected gestures on the frame
            for idx, gesture in enumerate(gesture_data + pose_data):
                cv2.putText(frame, gesture, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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

        summary_file.write("\nArm Movements Summary:\n")
        for arm_movement, count in arm_movement_counter.items():
            summary_file.write(f"{arm_movement}: {count}\n")

        summary_file.write("\nActivities Summary:\n")
        for activity, count in activity_counter.items():
            summary_file.write(f"{activity}: {count}\n")

    print(f"Summary written to {summary_path}")

# Run the detection function
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video.mp4')
output_video_path = os.path.join(script_dir, 'output_video.mp4')
summary_path = os.path.join(script_dir, 'summary.txt')

detect_features(input_video_path, output_video_path, summary_path)
