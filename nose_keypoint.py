import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def main():
    # Path to the downloaded Face Landmarker model
    model_path = "face_landmarker.task"

    # Configure the FaceLandmarker for VIDEO mode
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        # Open the webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Nose Keypoint Detection is running. Press 'ESC' to exit.")

        frame_timestamp_ms: int = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a mirror-like view
            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            # Convert BGR (OpenCV) to RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a MediaPipe Image from the numpy array
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Increment timestamp for VIDEO mode (must be monotonically increasing)
            frame_timestamp_ms += 33  # ~30 FPS

            # Run face landmark detection
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.face_landmarks:
                for face_landmarks in result.face_landmarks:
                    # Landmark index 1 = Nose tip
                    # (MediaPipe canonical face mesh: index 1 is the nose tip)
                    nose_tip = face_landmarks[1]

                    # Convert normalized coordinates to pixel coordinates
                    nose_x = int(nose_tip.x * w)
                    nose_y = int(nose_tip.y * h)

                    # Draw a filled red circle at the nose tip
                    cv2.circle(frame, (nose_x, nose_y), 10, (0, 0, 255), thickness=-1)

                    # Display coordinates on screen
                    cv2.putText(
                        frame,
                        f"Nose: ({nose_x}, {nose_y})",
                        (nose_x + 15, nose_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

            # Display the result
            cv2.imshow("Nose Keypoint Detection", frame)

            # Press ESC to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
