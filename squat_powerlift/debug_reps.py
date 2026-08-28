import cv2
import numpy as np
import matplotlib.pyplot as plt

from landmarks_squat import (
    extract_landmarks_from_video,
)


MODEL_PATH = "../modelo_mediapipe/pose_landmarker_full.task"
VIDEO_PATH = "videos_ejercicios/squat_frontal_2.mp4"

OUTPUT_VIDEO = "debug_repetitions.mp4"
OUTPUT_GRAPH = "debug_repetitions.png"


# Extraemos los landmarks separados por repeticiones
repetitions = extract_landmarks_from_video(
    VIDEO_PATH,
    MODEL_PATH
)


# Mostramos información sobre las repeticiones detectadas
print("\nInformación sobre las repeticiones detectadas:")

print(f"Repeticiones detectadas: {len(repetitions)}")

for i, rep in enumerate(repetitions):

    print(
        f"Repetición {i + 1}: "
        f"{len(rep)} frames"
    )


# Abrimos el vídeo original
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30


width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# Creamos el vídeo de salida
fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# Generamos una lista indicando a qué repetición pertenece cada frame
frame_rep = []


for i, rep in enumerate(repetitions):

    for _ in range(len(rep)):

        frame_rep.append(i + 1)


# Recorremos el vídeo y dibujamos la información
frame_idx = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break


    # Comprobamos si el frame pertenece a alguna repetición
    current_rep = None

    if frame_idx < len(frame_rep):

        current_rep = frame_rep[frame_idx]


    # Dibujamos la repetición actual
    if current_rep is not None:

        text = f"REPETICION {current_rep}"

        cv2.putText(
            frame,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    else:

        cv2.putText(
            frame,
            "FUERA DE REPETICION",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )


    # Mostramos el número de frame
    cv2.putText(
        frame,
        f"Frame: {frame_idx}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # Escribimos el frame en el vídeo de salida
    out.write(frame)

    frame_idx += 1


cap.release()
out.release()


print("\nArchivo de vídeo generado con la información de las repeticiones:")

print(f"Vídeo: {OUTPUT_VIDEO}")