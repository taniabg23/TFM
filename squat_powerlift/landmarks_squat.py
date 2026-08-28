import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

from scipy.signal import find_peaks

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions


# Calculamos el ángulo de la rodilla
def calculate_knee_angle(frame):

    landmarks = frame.reshape(33, 3)

    left_hip = landmarks[23]
    left_knee = landmarks[25]
    left_ankle = landmarks[27]

    right_hip = landmarks[24]
    right_knee = landmarks[26]
    right_ankle = landmarks[28]

    left_ba = left_hip - left_knee
    left_bc = left_ankle - left_knee

    right_ba = right_hip - right_knee
    right_bc = right_ankle - right_knee

    left_norm = np.linalg.norm(left_ba) * np.linalg.norm(left_bc)
    right_norm = np.linalg.norm(right_ba) * np.linalg.norm(right_bc)

    if left_norm == 0 or right_norm == 0:
        return np.nan

    left_cosine = np.dot(left_ba, left_bc) / left_norm
    right_cosine = np.dot(right_ba, right_bc) / right_norm

    left_cosine = np.clip(left_cosine, -1, 1)
    right_cosine = np.clip(right_cosine, -1, 1)

    left_angle = np.degrees(np.arccos(left_cosine))
    right_angle = np.degrees(np.arccos(right_cosine))

    return np.mean([left_angle, right_angle])


# Detectamos las repeticiones a partir de la señal de la rodilla
def detect_repetitions(signal):

    signal = np.asarray(signal, dtype=float)

    valid = ~np.isnan(signal)

    if np.sum(valid) < 30:
        return []

    original_indices = np.where(valid)[0]
    signal = signal[valid]

    # Suavizamos la señal para reducir el ruido
    smooth = pd.Series(signal).rolling(
        15,
        center=True
    ).mean()

    smooth = smooth.bfill().ffill().values

    # Buscamos los fondos de las sentadillas
    # El fondo corresponde al menor ángulo de rodilla
    bottoms, _ = find_peaks(
        -smooth,
        distance=30,
        prominence=8
    )

    if len(bottoms) == 0:
        return []

    reps = []

    # Recorremos todos los fondos detectados
    for i, bottom in enumerate(bottoms):

        # Definimos el inicio de la búsqueda
        # como el fondo anterior o el principio del vídeo
        if i == 0:
            search_start = 0
        else:
            search_start = bottoms[i - 1]

        # Definimos el final de la búsqueda
        # como el siguiente fondo o el final del vídeo
        if i == len(bottoms) - 1:
            search_end = len(smooth)
        else:
            search_end = bottoms[i + 1]

        # Buscamos el punto más alto antes del fondo
        before = smooth[search_start:bottom + 1]

        if len(before) == 0:
            continue

        start = search_start + np.argmax(before)

        # Buscamos el punto más alto después del fondo
        after = smooth[bottom:search_end]

        if len(after) == 0:
            continue

        end = bottom + np.argmax(after)

        # Comprobamos que existe una bajada
        if start >= bottom:
            continue

        # Comprobamos que existe una subida
        if end <= bottom:
            continue

        # Comprobamos que la repetición tiene una duración suficiente
        if (end - start) < 20:
            continue

        # Convertimos los índices de la señal
        # a índices originales del vídeo
        start_original = original_indices[start]
        end_original = original_indices[end]

        reps.append(
            (start_original, end_original)
        )

    return reps


# Extraemos los landmarks y dividimos el vídeo en repeticiones
def extract_landmarks_from_video(video_path, model_path):

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    detector = PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)

    sequence = []
    knee_signal = []

    frame_idx = 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp = int(frame_idx / fps * 1000)

        result = detector.detect_for_video(
            mp_image,
            timestamp
        )

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            frame_landmarks = []

            for lm in landmarks:

                frame_landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            frame_landmarks = np.array(
                frame_landmarks,
                dtype=np.float32
            )

            sequence.append(frame_landmarks)

            knee_signal.append(
                calculate_knee_angle(frame_landmarks)
            )

        else:

            sequence.append(
                np.zeros(99, dtype=np.float32)
            )

            knee_signal.append(
                np.nan
            )

        frame_idx += 1

    cap.release()
    detector.close()

    sequence = np.array(
        sequence,
        dtype=np.float32
    )

    knee_signal = np.array(
        knee_signal,
        dtype=np.float32
    )

    # Detectamos las repeticiones
    repetitions = detect_repetitions(knee_signal)

    print(f"Repeticiones detectadas: {len(repetitions)}")

    # Mostramos los límites de cada repetición
    for i, (start, end) in enumerate(repetitions):

        print(
            f"Repetición {i + 1}: "
            f"frame {start} -> frame {end} "
            f"({end - start} frames)"
        )

    # Separamos los landmarks correspondientes a cada repetición
    repetitions_sequence = []

    for start, end in repetitions:

        rep = sequence[start:end + 1]

        if len(rep) > 0:

            repetitions_sequence.append(
                np.asarray(
                    rep,
                    dtype=np.float32
                )
            )

    return repetitions_sequence