import pickle
import cv2
import mediapipe as mp
from utils import get_face_landmarks

# Danh sách cảm xúc theo đúng thứ tự khi train
emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Khởi tạo face_mesh chỉ một lần
with mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                     max_num_faces=1,
                                     min_detection_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Truyền face_mesh vào hàm
        face_landmarks = get_face_landmarks(frame, face_mesh, draw=True)

        if face_landmarks is not None:
            output = model.predict([face_landmarks])
            emotion_text = emotions[int(output[0])]
        else:
            emotion_text = "No face"

        cv2.putText(frame,
                    emotion_text,
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
