import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def get_face_landmarks(image, face_mesh, draw=False):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])  # 468 điểm × 3
        if draw:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS)
        return np.array(landmarks)
    return None
