import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model/fer2013_cnn.h5')
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img_dim = (48, 48)

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return frame, None

    # ✅ Chọn khuôn mặt lớn nhất (gần camera nhất)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    (x, y, w, h) = faces[0]

    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, img_dim)
    face = face / 255.0
    face = face.reshape(1, img_dim[0], img_dim[1], 1)

    pred = model.predict(face)
    emotion_idx = np.argmax(pred)
    emotion = emotions[emotion_idx]

    # Vẽ duy nhất 1 khung và nhãn cảm xúc
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, emotion, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, emotion
