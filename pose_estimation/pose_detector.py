import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect_pose(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks and draw:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        return frame, results

    def get_landmarks(self, results, frame_shape):
        landmarks = []

        if not results.pose_landmarks:
            return landmarks

        height, width, _ = frame_shape

        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks.append({
                "id": idx,
                "x": int(landmark.x * width),
                "y": int(landmark.y * height),
                "z": landmark.z,
                "visibility": landmark.visibility,
            })

        return landmarks