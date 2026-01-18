import cv2
import numpy as np
import time
import csv
from collections import deque, Counter
from tensorflow.keras.models import load_model

MODEL_PATH = "emotion_model.h5"

IMG_SIZE = 64

CAM_INDEX = 0

LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

SMOOTH_WINDOW = 12

TIMELINE_LEN = 120  

LOG_FILE = "emotion_log.csv"
LOG_EVERY_SECONDS = 1.0  

CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30

ENABLE_DISPLAY_TUNING = True
DISPLAY_ALPHA = 1.15  
DISPLAY_BETA = 25     

ENABLE_UNSHARP_MASK = True
UNSHARP_SIGMA = 1.2
UNSHARP_AMOUNT = 0.35  

DISABLE_AUTOFOCUS = True
MANUAL_FOCUS_VALUE = 80  

AUTO_EXPOSURE_MODE = True

model = load_model(MODEL_PATH, compile=False)
print("Model input shape:", model.input_shape)

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

pred_buffer = deque(maxlen=SMOOTH_WINDOW)
timeline = deque(maxlen=TIMELINE_LEN)

with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "emotion", "confidence"])

last_log_time = 0.0

def majority_vote(items):
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]

def tune_display(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Make the display brighter/clearer without messing with model input.
    """
    out = frame_bgr

    if ENABLE_DISPLAY_TUNING:
        out = cv2.convertScaleAbs(out, alpha=DISPLAY_ALPHA, beta=DISPLAY_BETA)

    if ENABLE_UNSHARP_MASK:
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=UNSHARP_SIGMA)
        # out = out*(1+amount) - blur*amount
        out = cv2.addWeighted(out, 1.0 + UNSHARP_AMOUNT, blur, -UNSHARP_AMOUNT, 0)

    return out

def draw_timeline(frame, timeline_data, labels):
    """Draw a simple timeline bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 90
    y0 = h - bar_h - 10
    x0 = 10
    x1 = w - 10

    cv2.rectangle(frame, (x0, y0), (x1, y0 + bar_h), (20, 20, 20), -1)

    if not timeline_data:
        cv2.putText(frame, "Timeline: (no data yet)", (x0 + 10, y0 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n_classes = len(labels)

    points = []
    for i, emo in enumerate(timeline_data):
        xi = int(np.interp(i, [0, len(timeline_data) - 1], [x0 + 10, x1 - 10]))
        yi = int(np.interp(label_to_idx.get(emo, 0),
                           [0, n_classes - 1],
                           [y0 + bar_h - 10, y0 + 10]))
        points.append((xi, yi))

    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (180, 180, 180), 2)

    cv2.putText(frame, "Emotion timeline (recent)", (x0 + 10, y0 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    cv2.putText(frame, f"Last: {timeline_data[-1]}", (x0 + 10, y0 + bar_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

def preprocess_face(gray_face):
    """Resize to model input size and format as (1, IMG_SIZE, IMG_SIZE, 1)."""
    face_resized = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
    face_input = face_resized.astype("float32") / 255.0
    face_input = np.expand_dims(face_input, axis=-1)  # (H,W,1)
    face_input = np.expand_dims(face_input, axis=0)   # (1,H,W,1)
    return face_input

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try CAM_INDEX=1 or CAM_INDEX=2.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

if DISABLE_AUTOFOCUS:
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, MANUAL_FOCUS_VALUE)

if AUTO_EXPOSURE_MODE:
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

print("Camera capture:",
      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
      "@", int(cap.get(cv2.CAP_PROP_FPS)), "fps")

print("Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t_now = time.time()
    display_frame = tune_display(frame.copy())

    gray_for_detect = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_for_detect,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(90, 90)
    )

    chosen = None
    if len(faces) > 0:
        chosen = max(faces, key=lambda r: r[2] * r[3])

    if chosen is not None:
        x, y, w, h = chosen

        pad = int(0.12 * w)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(display_frame.shape[1], x + w + pad)
        y1 = min(display_frame.shape[0], y + h + pad)

        face_roi_gray = gray_for_detect[y0:y1, x0:x1]
        face_input = preprocess_face(face_roi_gray)

        preds = model.predict(face_input, verbose=0)[0]
        emo_raw = LABELS[int(np.argmax(preds))]
        conf_raw = float(np.max(preds))

        pred_buffer.append(emo_raw)
        emo_smooth = majority_vote(pred_buffer)
        timeline.append(emo_smooth)

        if (t_now - last_log_time) >= LOG_EVERY_SECONDS:
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), emo_smooth, f"{conf_raw:.3f}"])
            last_log_time = t_now

        cv2.rectangle(display_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label_text = f"{emo_smooth} (conf {conf_raw:.2f})"
        cv2.putText(display_frame, label_text, (x0, max(30, y0 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

    draw_timeline(display_frame, list(timeline), LABELS)

    cv2.imshow("Emotion Detection (Local CNN)", display_frame)

    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved log to: {LOG_FILE}")
