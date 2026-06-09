import cv2
from pose_estimation.pose_detector import PoseDetector
from utils.fps_utils import FPSCounter
from utils.angle_utils import extract_squat_features


def main():
    cap = cv2.VideoCapture(-1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = PoseDetector()
    fps_counter = FPSCounter()

    while True:
        success, frame = cap.read()

        if not success:
            print("Error: Failed to read frame.")
            break

        frame, results = detector.detect_pose(frame, draw=True)
        landmarks = detector.get_landmarks(results, frame.shape)

        features = extract_squat_features(landmarks)
        
        if features:
            y = 120

            for key, value in features.items():
                cv2.putText(
                    frame,
                    f"{key}: {value}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )
                y += 25

        else:
            cv2.putText(
                frame,
                "Show full body for squat feature extraction",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        
        fps = fps_counter.get_fps()

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Landmarks Detected: {len(landmarks)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow("PhysioPose - Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()