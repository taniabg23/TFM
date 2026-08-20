import numpy as np
import tensorflow as tf

from landmarks_squat import extract_landmarks_from_video


MODEL_PATH = "../modelo_mediapipe/pose_landmarker_full.task"
VIDEO_PATH = "videos_ejercicios/squat_frontal_2.mp4"

MAX_LEN = 60
PAD_VALUE = -999.0
THRESHOLD = 0.4
STRIDE = 10


# Cargamos el modelo entrenado
model = tf.keras.models.load_model("squat_model.keras")

# Cargamos estadísticas de estandarización del entrenamiento
stats = np.load("standardizer_stats.npz")
mean = stats["mean"].astype(np.float32)
std = stats["std"].astype(np.float32)
MAX_LEN = int(stats["max_len"][0])
PAD_VALUE = float(stats["pad_value"][0])


# Extraemos los landmarks del video 
sequence = extract_landmarks_from_video(
    VIDEO_PATH,
    MODEL_PATH
)

sequence = np.array(sequence, dtype=np.float32)

if len(sequence) == 0:
    raise ValueError("No se pudieron extraer landmarks del video.")


flat_seq = sequence.reshape(len(sequence), -1)
flat_seq = (flat_seq - mean) / std

windows = []

# Para videos cortos, una sola ventana con padding
if len(flat_seq) <= MAX_LEN:
    window = flat_seq
    if len(window) < MAX_LEN:
        pad = np.full((MAX_LEN - len(window), 99), PAD_VALUE, dtype=np.float32)
        window = np.concatenate([window, pad], axis=0)
    windows.append(window)
else:
    # Para videos largos, evaluamos varias ventanas para no depender de los primeros frames
    for start in range(0, len(flat_seq) - MAX_LEN + 1, STRIDE):
        windows.append(flat_seq[start:start + MAX_LEN])

    # Asegurar una ventana final pegada al final del video
    last_start = len(flat_seq) - MAX_LEN
    if last_start % STRIDE != 0:
        windows.append(flat_seq[last_start:last_start + MAX_LEN])

X = np.array(windows, dtype=np.float32)

# Prediccion por ventana
preds = model.predict(X, verbose=0).reshape(-1)
pred_max = float(np.max(preds))
pred_mean = float(np.mean(preds))

print(f"Frames totales: {len(sequence)}")
print(f"Ventanas evaluadas: {len(windows)}")
print(f"PRED_MAX: {pred_max:.4f}")
print(f"PRED_MEAN: {pred_mean:.4f}")
print(f"THRESHOLD: {THRESHOLD}")

if pred_max > THRESHOLD:
    print("VALID SQUAT")
else:
    print("INVALID SQUAT")