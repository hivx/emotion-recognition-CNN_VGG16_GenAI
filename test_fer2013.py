import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load mô hình đã huấn luyện
model = load_model('vgg16_cnn.h5')

# Định nghĩa nhãn cảm xúc (theo FER2013)
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Load Haar Cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Khởi tạo camera
cap = cv2.VideoCapture(0)  # 0 là webcam mặc định

# Kích thước ảnh đầu vào mô hình
img_dim = (48, 48)

while True:
    ret, frame = cap.read()  # Đọc frame từ camera
    if not ret:
        break

    # Chuyển frame sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Cắt khuôn mặt từ ảnh xám
        face = gray[y:y+h, x:x+w]
        # Resize về 48x48
        face = cv2.resize(face, img_dim)
        # Chuẩn hóa
        face = face / 255.0
        # Reshape để phù hợp với đầu vào mô hình
        face = face.reshape(1, img_dim[0], img_dim[1], 1)

        # Dự đoán cảm xúc
        pred = model.predict(face)
        emotion_idx = np.argmax(pred)
        emotion = emotions[emotion_idx]

        # Vẽ khung và nhãn cảm xúc lên frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow('Emotion Detection', frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()