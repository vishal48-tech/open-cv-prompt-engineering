import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def calculate_angle(a, b, c):
    """
    Calculate the angle at point B formed by points A -> B -> C.
    Each point is a tuple (x, y).
    Returns the angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b  # Vector from B to A
    bc = c - b  # Vector from B to C

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def draw_glass_overlay(frame, text, position, font_scale=1.2, thickness=3):
    """
    Draw text with a semi-transparent black glass background.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    # Rectangle coordinates with padding
    pad_x, pad_y = 30, 20
    x, y = position
    x1 = x - pad_x
    y1 = y - text_h - pad_y
    x2 = x + text_w + pad_x
    y2 = y + baseline + pad_y

    # Create the semi-transparent black glass overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Add a subtle border for the glass effect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 2)

    # Draw the text
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)


def main():
    # ── MediaPipe tasks API Setup ─────────────────────────────────────
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        # ── Angle threshold (degrees) ────────────────────────────────────
        # If ear-shoulder-hip angle < this, the person is slouching
        GOOD_POSTURE_THRESHOLD = 160

        # ── Webcam ───────────────────────────────────────────────────────
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Posture Detection is running (side view).")
        print("Sit sideways to the camera so it sees your profile.")
        print("Press 'ESC' to exit.")

        frame_timestamp_ms: int = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            h, w, _ = frame.shape

            # Convert BGR → RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            frame_timestamp_ms += 33
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if results.pose_landmarks:
                # We requested num_poses=1, so take the first pose
                landmarks = results.pose_landmarks[0]

                # ── Pick the side with higher visibility ─────────────
                # Left side landmarks: 7 (ear), 11 (shoulder), 23 (hip)
                left_ear = landmarks[7]
                left_shoulder = landmarks[11]
                left_hip = landmarks[23]

                # Right side landmarks: 8 (ear), 12 (shoulder), 24 (hip)
                right_ear = landmarks[8]
                right_shoulder = landmarks[12]
                right_hip = landmarks[24]

                # Use the side with higher average visibility
                left_vis = (left_ear.visibility + left_shoulder.visibility + left_hip.visibility) / 3
                right_vis = (right_ear.visibility + right_shoulder.visibility + right_hip.visibility) / 3

                if left_vis >= right_vis:
                    ear, shoulder, hip = left_ear, left_shoulder, left_hip
                else:
                    ear, shoulder, hip = right_ear, right_shoulder, right_hip

                # Convert normalized coordinates to pixel coordinates
                ear_px = (int(ear.x * w), int(ear.y * h))
                shoulder_px = (int(shoulder.x * w), int(shoulder.y * h))
                hip_px = (int(hip.x * w), int(hip.y * h))

                # ── Calculate angle at the shoulder ──────────────────
                angle = calculate_angle(ear_px, shoulder_px, hip_px)

                # ── Draw the 3 keypoints and connecting lines ────────
                # Lines
                cv2.line(frame, ear_px, shoulder_px, (255, 200, 0), 3)
                cv2.line(frame, shoulder_px, hip_px, (255, 200, 0), 3)

                # Points
                cv2.circle(frame, ear_px, 8, (0, 255, 255), -1)
                cv2.circle(frame, shoulder_px, 8, (0, 255, 255), -1)
                cv2.circle(frame, hip_px, 8, (0, 255, 255), -1)

                # ── Display angle near the shoulder ──────────────────
                cv2.putText(
                    frame,
                    f"{int(angle)} deg",
                    (shoulder_px[0] + 15, shoulder_px[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 200, 0),
                    2,
                )

                # ── Posture decision ─────────────────────────────────
                if angle < GOOD_POSTURE_THRESHOLD:
                    # BAD POSTURE — show warning with black glass overlay
                    draw_glass_overlay(
                        frame,
                        "Sit Straight!",
                        position=(w // 2 - 150, 60),
                        font_scale=1.5,
                        thickness=3,
                    )
                else:
                    # GOOD POSTURE
                    cv2.putText(
                        frame,
                        "Good Posture",
                        (w // 2 - 100, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3,
                    )

            else:
                # No pose detected
                cv2.putText(
                    frame,
                    "No pose detected — sit sideways to camera",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # ── Show frame ───────────────────────────────────────────────
            cv2.imshow("Posture Detection (Side View)", frame)

            # Press ESC to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
