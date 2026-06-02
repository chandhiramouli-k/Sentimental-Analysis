import cv2
from tensorflow.keras.models import load_model
from config import MODEL_PATH
from src.face_detection import detect_faces
from src.sentiment_mapping import emotion_to_sentiment
from src.utils import preprocess_face, predict_emotion, draw_label

model = load_model(MODEL_PATH)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray, faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        roi = preprocess_face(gray, x, y, w, h)
        emotion, confidence = predict_emotion(model, roi)
        sentiment = emotion_to_sentiment(emotion)

        draw_label(frame, x, y, w, h,
                   emotion, sentiment, confidence)

    cv2.imshow("Live Sentiment Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()