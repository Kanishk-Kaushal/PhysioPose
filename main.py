import cv2
from pose_estimation.pose_detector import PoseDetector
from utils.fps_utils import FPSCounter


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