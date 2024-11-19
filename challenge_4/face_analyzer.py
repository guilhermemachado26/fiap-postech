import os
from collections import Counter
from concurrent import futures

import cv2
import mediapipe as mp
import numpy as np

from deepface import DeepFace
from tqdm import tqdm
from scipy.spatial.distance import euclidean


# Initialize MediaPipe for gesture detection
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class VideoAnalyzer:
    def __init__(self):
        self.total_anomalies = 0
        self.emotion_counter = Counter()
        self.arm_movement_counter = Counter()
        self.activity_counter = Counter()

    def run(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_video_path = os.path.join(script_dir, "video.mp4")
        output_video_path = os.path.join(script_dir, "output_video.mp4")
        summary_path = os.path.join(script_dir, "summary.txt")

        self.analyze_video(input_video_path, output_video_path, summary_path)

    def analyze_frame(self, frame):
        # Analyze frame for face recognition and emotions
        results = DeepFace.analyze(frame, actions=["emotion", "age", "gender"], enforce_detection=False)
        face_data = []
        for face in results:
            x, y, w, h = (face["region"]["x"], face["region"]["y"], face["region"]["w"], face["region"]["h"])
            dominant_gender = max(face["gender"], key=face["gender"].get)  # Determine dominant gender
            face_data.append({"box": (x, y, w, h), "dominant_emotion": face["dominant_emotion"], "age": face.get("age"), "gender": dominant_gender})
        return face_data

    def analyze_face_emotions(self, frame):
        face_data = self.analyze_frame(frame)
        for face in face_data:
            x, y, w, h = face["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            dominant_emotion = face["dominant_emotion"]
            self.emotion_counter[dominant_emotion] += 1
            cv2.putText(frame, f"{dominant_emotion} | Gender: {face['gender']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def detect_handshake(self, hands, frame):
        """Detect handshake gesture using MediaPipe Hands."""
        handshake_detected = False
        hand_landmarks_list = []
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_landmarks_list.append(hand_landmarks)

            # Check if at least two hands are detected
            if len(hand_landmarks_list) >= 2:
                # Compute distance between wrists of two hands
                left_hand = hand_landmarks_list[0].landmark[mp_hands.HandLandmark.WRIST]
                right_hand = hand_landmarks_list[1].landmark[mp_hands.HandLandmark.WRIST]

                distance = euclidean((left_hand.x, left_hand.y), (right_hand.x, right_hand.y))

                # Define handshake threshold for proximity
                if distance < 0.1:  # Adjust this threshold for your scenario
                    handshake_detected = True

        return handshake_detected

    def analyze_gestures(self, frame, hand_detector, pose_detector, prev_pose_landmarks, frame_count):
        gesture_data = []
        hands = hand_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        handshake_detected = self.detect_handshake(hand_detector, frame)

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
            left_arm_movement = (
                abs(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y - prev_pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y)
                if prev_pose_landmarks
                else 0
            )
            right_arm_movement = (
                abs(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y - prev_pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y)
                if prev_pose_landmarks
                else 0
            )
            left_leg_movement = left_ankle_movement
            right_leg_movement = right_ankle_movement

            # Thresholds for detecting dancing (simultaneous arm and leg movements)
            if (left_arm_movement > 0.05 and right_arm_movement > 0.05) and (left_leg_movement > 0.02 and right_leg_movement > 0.02):
                pose_data.append("Dancing")

            # Add handshake detection result
            if handshake_detected:
                pose_data.append("Handshake")

            # Update previous landmarks for the next frame
            prev_pose_landmarks = landmarks

        # Increment gesture counts
        for gesture in gesture_data:
            self.activity_counter[gesture] += 1
        for pose in pose_data:
            if "Arm" in pose:
                self.arm_movement_counter[pose] += 1
            elif pose in ["Walking", "Dancing", "Handshake"]:
                self.activity_counter[pose] += 1

        # Display detected gestures on the frame
        for idx, gesture in enumerate(gesture_data + pose_data):
            cv2.putText(frame, gesture, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def analyze_abrupt_movement(self, prev_frame, curr_frame, threshold=100000):
        # Calculate the difference between frames and detect abrupt motion
        if prev_frame is None:
            return

        diff = cv2.absdiff(prev_frame, curr_frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_score = np.sum(diff_gray)

        if diff_score > threshold:
            self.total_anomalies += 1
            cv2.putText(curr_frame, "Anomaly: Abrupt Movement", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def analyze_video(self, video_path, output_path, summary_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_skip = 10
        prev_frame = None
        prev_pose_landmarks = None

        with mp_hands.Hands() as hand_detector, mp_pose.Pose() as pose_detector:
            for frame_count in tqdm(range(total_frames), desc="Processing video"):
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    self.analyze_abrupt_movement(prev_frame, frame)
                    self.analyze_face_emotions(frame)
                    self.analyze_gestures(frame, hand_detector, pose_detector, prev_pose_landmarks, frame_count)

                    prev_frame = frame

                out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Write summary to text file
        with open(summary_path, "w") as summary_file:
            summary_file.write(f"Total Frames Analyzed: {total_frames}\n")
            summary_file.write(f"Total Anomalies Detected: {self.total_anomalies}\n\n")

            summary_file.write("Emotion Summary:\n")
            for emotion, count in self.emotion_counter.items():
                summary_file.write(f"{emotion}: {count}\n")

            summary_file.write("\nArm Movements Summary:\n")
            for arm_movement, count in self.arm_movement_counter.items():
                summary_file.write(f"{arm_movement}: {count}\n")

            summary_file.write("\nActivities Summary:\n")
            for activity, count in self.activity_counter.items():
                summary_file.write(f"{activity}: {count}\n")

        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    analyzer = VideoAnalyzer()
    analyzer.run()
