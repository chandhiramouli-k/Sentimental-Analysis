import cv2
import numpy as np
from config import IMG_SIZE, EMOTIONS

def preprocess_face(gray_frame, x, y, w, h):
    """
    Extracts and preprocesses face ROI
    """
    roi = gray_frame[y:y+h, x:x+w]
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi / 255.0
    roi = roi.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return roi


def predict_emotion(model, roi):
    """
    Predict emotion and confidence
    """
    prediction = model.predict(roi, verbose=0)
    emotion_index = np.argmax(prediction)
    emotion = EMOTIONS[emotion_index]
    confidence = float(np.max(prediction))
    return emotion, confidence


def draw_label(frame, x, y, w, h, emotion, sentiment, confidence):
    """
    Draw bounding box and label
    """
    cv2.rectangle(frame, (x, y), (x+w, y+h),
                  (0, 255, 0), 2)

    label = f"{emotion} ({confidence:.2f}) | {sentiment}"

    cv2.putText(frame,
                label,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2)