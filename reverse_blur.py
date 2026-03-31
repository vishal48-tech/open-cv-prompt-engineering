import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
    # Path to the downloaded selfie segmentation model
    model_path = "selfie_segmentation_landscape.tflite"

    # Configure the ImageSegmenter with confidence masks enabled
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False
    )

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        # Open the webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Reverse Blur Filter is running. Press 'ESC' to exit.")

        frame_timestamp_ms: int = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a mirror-like view
            frame = cv2.flip(frame, 1)

            # Convert BGR (OpenCV) to RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a MediaPipe Image from the numpy array
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Increment timestamp for VIDEO mode (must be monotonically increasing)
            frame_timestamp_ms += 33  # ~30 FPS

            # Run segmentation
            result = segmenter.segment_for_video(mp_image, frame_timestamp_ms)

            # Get the confidence mask for the person (index 0)
            confidence_mask = result.confidence_masks[0].numpy_view()

            # Squeeze any extra dimensions so mask is (H, W)
            confidence_mask = np.squeeze(confidence_mask)

            # Create a boolean mask: True where the person is detected
            person_mask = np.stack((confidence_mask,) * 3, axis=-1) > 0.5

            # Create a heavily blurred version of the frame
            blurred_frame = cv2.GaussianBlur(frame, (99, 99), 0)

            # Reverse blur composite:
            #   - Subject (person_mask=True)  -> blurred pixels
            #   - Background (person_mask=False) -> original clear pixels
            output = np.where(person_mask, blurred_frame, frame)

            # Display the result
            cv2.imshow("Reverse Blur Filter (Subject Blur | Clear Background)", output)

            # Press ESC to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
