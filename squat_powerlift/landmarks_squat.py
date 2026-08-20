import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions


# Función para extraer los landmarks de un video utilizando MediaPipe
def extract_landmarks_from_video(video_path, model_path):

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    detector = PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)

    sequence = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        result = detector.detect_for_video(mp_image, frame_idx)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            frame_landmarks = []
            for lm in landmarks:
                frame_landmarks.extend([lm.x, lm.y, lm.z])

            sequence.append(frame_landmarks)
        else:
            sequence.append([0] * 99)

        frame_idx += 1

    cap.release()
    detector.close()

    return np.array(sequence)