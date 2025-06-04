import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

model = load_model('model_optimal.h5')

mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            # Lấy bbox face đầu tiên
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x1 = int(bboxC.xmin * iw)
            y1 = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Cắt vùng face
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = x1 + w, y1 + h
            face_img = frame[y1:y2, x1:x2]

            if face_img.size != 0:
                # Resize về đúng input CNN (vd 100x100)
                face_img = cv2.resize(face_img, (100, 100))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = face_img.astype('float32') / 255.0
                input_data = np.expand_dims(face_img, axis=0)  # (1,100,100,3)

                prediction = model.predict(input_data, verbose=0)
                predicted_emotion = emotions[np.argmax(prediction)]

                # Vẽ bbox và tên cảm xúc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                predicted_emotion = "No face"
        else:
            predicted_emotion = "No face"

        cv2.putText(frame, predicted_emotion, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Recognition (CNN)', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
