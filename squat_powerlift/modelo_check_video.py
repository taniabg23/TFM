import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "../modelo_mediapipe/pose_landmarker_full.task"

# Conexiones principales del cuerpo
POSE_CONNECTIONS = [

    # Cara
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),

    # Torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),

    # Brazo izquierdo
    (11, 13),
    (13, 15),

    # Brazo derecho
    (12, 14),
    (14, 16),

    # Pierna izquierda
    (23, 25),
    (25, 27),

    # Pierna derecha
    (24, 26),
    (26, 28)
]


def draw_landmarks(frame, landmarks):

    h, w, _ = frame.shape

    # Dibujar puntos
    for lm in landmarks:

        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    # Dibujar conexiones
    for start_idx, end_idx in POSE_CONNECTIONS:

        start = landmarks[start_idx]
        end = landmarks[end_idx]

        x1 = int(start.x * w)
        y1 = int(start.y * h)

        x2 = int(end.x * w)
        y2 = int(end.y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return frame


def test_pose(video_path):

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_idx = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]

                frame = draw_landmarks(frame, landmarks)

            cv2.imshow("Pose Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()


# ejemplo
test_pose("videos_ejercicios/squat_frontal_6.mp4")