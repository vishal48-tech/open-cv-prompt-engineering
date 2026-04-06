"""
detect_realtime.py
──────────────────
Real-time sign language detection using webcam.

Pipeline:
  1. Capture frame from webcam
  2. Detect hand using MediaPipe Tasks API
  3. Extract `hand_world_landmarks` (3D coordinates in meters)
  4. Compute **Pairwise Distances** for scale & rotation invariance! 
  5. Classify with our deep Residual MLP
  6. Display predicted letter with UI overlay

Usage:
    python detect_realtime.py
"""

import os
import time
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Configuration ────────────────────────────────────────────────────────────
DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DATA_ROOT, "sign_language_model.pth")
HAND_MODEL_PATH = os.path.join(DATA_ROOT, "hand_landmarker.task")

NUM_CLASSES = 26
NUM_KEYPOINTS = 21
FEATURES_PER_KP = 3
NUM_DISTANCES = (NUM_KEYPOINTS * (NUM_KEYPOINTS - 1)) // 2  # 210
INPUT_DIM = (NUM_KEYPOINTS * FEATURES_PER_KP) + NUM_DISTANCES  # 63 + 210 = 273

VALID_LETTERS = sorted("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IDX_TO_LETTER = {i: letter for i, letter in enumerate(VALID_LETTERS)}

COLOR_TEXT = (0, 255, 120)
COLOR_FPS = (100, 100, 255)
COLOR_LANDMARK = (0, 200, 255)
COLOR_CONNECTION = (50, 150, 50)
COLOR_PANEL = (40, 40, 40)
COLOR_CONF = (255, 200, 50)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


# ── Feature Extraction ───────────────────────────────────────────────────────
def extract_features(kps_flat: np.ndarray) -> np.ndarray:
    """Extract 63 raw coords + 210 pairwise distances."""
    kps = kps_flat.reshape(NUM_KEYPOINTS, 3)
    
    # Calculate all pairwise distances
    distances = []
    for i in range(NUM_KEYPOINTS):
        for j in range(i + 1, NUM_KEYPOINTS):
            dist = np.linalg.norm(kps[i] - kps[j])
            distances.append(dist)
            
    # Concatenate original coordinates and distances
    return np.concatenate([kps_flat, np.array(distances)])


# ── Model ────────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, size, dropout=0.3):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return x + self.layer(x)

class SignLanguageResNet(nn.Module):
    """Residual Network matching the training architecture."""
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        
        self.res1 = ResidualBlock(512, 0.3)
        self.res2 = ResidualBlock(512, 0.3)
        
        self.output_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.output_layer(x)


# ── UI & Drawing ─────────────────────────────────────────────────────────────
def draw_landmarks(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for s, e in HAND_CONNECTIONS:
        if s < len(points) and e < len(points):
            cv2.line(frame, points[s], points[e], COLOR_CONNECTION, 2)
    for i, pt in enumerate(points):
        cv2.circle(frame, pt, 5 if i == 0 else 3, COLOR_LANDMARK, -1)

def draw_ui(frame, letter, confidence, fps, hand_detected):
    h, w = frame.shape[:2]
    panel_w = 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    px = w - panel_w + 15

    cv2.putText(frame, "SIGN LANGUAGE", (px, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    cv2.putText(frame, "DETECTOR", (px, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    cv2.line(frame, (px, 75), (w - 15, 75), (80, 80, 80), 1)

    if hand_detected and letter:
        cv2.putText(frame, letter, (px + 30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, COLOR_TEXT, 6)
        cv2.putText(frame, f"Conf: {confidence:.0%}", (px, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CONF, 1)
        bar_w = int(180 * confidence)
        cv2.rectangle(frame, (px, 250), (px + 180, 265), (60, 60, 60), -1)
        cv2.rectangle(frame, (px, 250), (px + bar_w, 265), COLOR_CONF, -1)
    else:
        cv2.putText(frame, "Show hand", (px, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(frame, "to detect", (px, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (px, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FPS, 1)
    cv2.putText(frame, "Press 'q' to quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    return frame

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index (e.g. 1 for OBS Virtual Camera)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Sign Language — WORLD Landmarks Real-Time Detection")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ Model not found: {MODEL_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = SignLanguageResNet(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  ✅ Classifier loaded ({device})")

    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    print("  ✅ Hand landmarker loaded")

    print(f"  📷 Attempting to open camera index {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"  ❌ Cannot open webcam at index {args.camera}. Try --camera 1 or --camera 2")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"  ✅ Webcam ({args.camera}) opened — press 'q' to quit\n")

    prev_time = time.time()
    fps = 0
    prediction_buffer = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, ts)

            letter = None
            confidence = 0.0
            hand_detected = False

            if result.hand_landmarks and result.hand_world_landmarks:
                hand_detected = True
                
                draw_landmarks(frame, result.hand_landmarks[0], w, h)

                kps = []
                for lm in result.hand_world_landmarks[0]:
                    kps.extend([lm.x, lm.y, lm.z])
                
                # IMPORTANT: Extract features! (Pairwise distances)
                kps_flat = np.array(kps, dtype=np.float32)
                features = extract_features(kps_flat)
                
                x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(x_tensor)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = probs.max(1)
                    confidence = conf.item()
                    raw_letter = IDX_TO_LETTER[pred_idx.item()]

                prediction_buffer.append(raw_letter)
                if len(prediction_buffer) > 5:
                    prediction_buffer.pop(0)
                letter = Counter(prediction_buffer).most_common(1)[0][0]
            else:
                prediction_buffer.clear()

            now = time.time()
            fps = 0.8 * fps + 0.2 / max(now - prev_time, 1e-6)
            prev_time = now

            frame = draw_ui(frame, letter, confidence, fps, hand_detected)
            cv2.imshow("Sign Language Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("\n  👋 Detection stopped.")

if __name__ == "__main__":
    main()
