import os
import cv2
import numpy as np
from utils import get_face_landmarks
import mediapipe as mp

data_dir = 'data'
output = []

# Danh sách cảm xúc theo thứ tự bạn định nghĩa
emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                     max_num_faces=1,
                                     min_detection_confidence=0.5) as face_mesh:
    for emotion_indx, emotion in enumerate(emotions):
        emotion_path = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_path):
            print(f"⚠️ Bỏ qua thư mục không tồn tại: {emotion_path}")
            continue

        for image_path_ in os.listdir(emotion_path):
            image_path = os.path.join(emotion_path, image_path_)
            image = cv2.imread(image_path)

            if image is None:
                print(f"⚠️ Không đọc được ảnh: {image_path}")
                continue

            face_landmarks = get_face_landmarks(image, face_mesh)

            if face_landmarks is not None and len(face_landmarks) == 1404:
                face_landmarks.append(int(emotion_indx))
                output.append(face_landmarks)
            else:
                print(f"❌ Bỏ qua ảnh (landmark không hợp lệ): {image_path}")

if output:
    np.savetxt('data.txt', np.asarray(output), fmt='%.6f', delimiter=',')
    print("✅ Đã lưu data.txt với", len(output), "mẫu.")
else:
    print("⚠️ Không có ảnh nào hợp lệ để lưu.")
