import numpy as np
import tensorflow as tf

from landmarks_squat import extract_landmarks_from_video


MODEL_PATH = "../modelo_mediapipe/pose_landmarker_full.task"
VIDEO_PATH = "videos_ejercicios/squat_frontal_2.mp4"

MAX_LEN = 60


# Cargamos el modelo entrenado
model = tf.keras.models.load_model("./trains/squat_model.h5")


# Extraemos los landmarks separados por repeticiones
repetitions = extract_landmarks_from_video(
    VIDEO_PATH,
    MODEL_PATH
)


# Evaluamos cada repetición por separado
valid_count = 0
invalid_count = 0


for i, rep in enumerate(repetitions):

    print(f"\nEvaluando repetición {i + 1}...")

    # Reshapeamos la repetición para que tenga la forma (num_frames, num_landmarks*3)
    seq = rep.reshape(len(rep), -1).astype(np.float32)

    # Ajustamos la longitud de la repetición
    if len(seq) > MAX_LEN:

        seq = seq[:MAX_LEN]

    else:

        pad = np.zeros(
            (MAX_LEN - len(seq), 99),
            dtype=np.float32
        )

        seq = np.concatenate(
            [seq, pad],
            axis=0
        )

    # Añadimos la dimensión correspondiente al batch
    X = np.expand_dims(
        seq,
        axis=0
    ).astype(np.float32)

    # Hacemos la predicción con el modelo
    pred = model.predict(
        X,
        verbose=0
    )[0][0]

    if pred > 0.5:

        print(f"Repetición {i + 1}: VALID SQUAT")
        print(f"Probabilidad: {pred:.4f}")

        valid_count += 1

    else:

        print(f"Repetición {i + 1}: INVALID SQUAT")
        print(f"Probabilidad: {pred:.4f}")

        invalid_count += 1


# Mostramos el resumen final
print("\nEvaluación:")

print(f"Repeticiones evaluadas: {len(repetitions)}")
print(f"Valid squat: {valid_count}")
print(f"Invalid squat: {invalid_count}")