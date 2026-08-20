import numpy as np
import tensorflow as tf

from landmarks_squat import extract_landmarks_from_video


MODEL_PATH = "../modelo_mediapipe/pose_landmarker_full.task"
VIDEO_PATH = "videos_ejercicios/squat_frontal_3.mp4"

MAX_LEN = 60


# Cargamos el modelo entrenado
model = tf.keras.models.load_model("squat_model.keras")


# Extraemos los landmarks del video 
sequence = extract_landmarks_from_video(
    VIDEO_PATH,
    MODEL_PATH
)

sequence = np.array(sequence)


# Reshapeamos la secuencia para que tenga la forma (num_frames, num_landmarks*3)
seq = sequence.reshape(len(sequence), -1)

if len(seq) > MAX_LEN:
    seq = seq[:MAX_LEN]
else:
    pad = np.zeros((MAX_LEN - len(seq), 99))
    seq = np.concatenate([seq, pad], axis=0)


X = np.expand_dims(seq, axis=0)


# Hacemos la predicción con el modelo
pred = model.predict(X)[0][0]

print("PRED:", pred)

if pred > 0.5:
    print("VALID SQUAT")
else:
    print("INVALID SQUAT")