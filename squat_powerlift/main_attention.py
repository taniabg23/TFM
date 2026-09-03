import numpy as np
import tensorflow as tf

from landmarks_squat import extract_landmarks_from_video


MODEL_PATH = "../modelo_mediapipe/pose_landmarker_full.task"
VIDEO_PATH = "videos_ejercicios/squat_frontal_1.mp4"

MAX_LEN = 60


# Definimos la capa Attention utilizada durante el entrenamiento
class Attention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )

        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super(Attention, self).build(input_shape)

    def call(self, x):

        e = tf.tanh(
            tf.matmul(x, self.W) + self.b
        )

        e = tf.squeeze(
            e,
            axis=-1
        )

        alpha = tf.nn.softmax(
            e,
            axis=1
        )

        context = tf.reduce_sum(
            x * tf.expand_dims(alpha, -1),
            axis=1
        )

        return context


# Cargamos el modelo entrenado
model = tf.keras.models.load_model(
    "./squat_model_best.keras",
    custom_objects={
        "Attention": Attention
    }
)


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
    seq = rep.reshape(
        len(rep),
        -1
    ).astype(np.float32)

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

        # Calculamos los errores de la repetición
        errors = []

        # Guardamos las variables biomecánicas de cada frame
        left_knee_angles = []
        right_knee_angles = []

        left_knee_x = []
        right_knee_x = []

        left_ankle_x = []
        right_ankle_x = []

        left_hip_y = []
        right_hip_y = []

        left_hip_x = []
        right_hip_x = []

        torso_angles = []

        # Analizamos cada frame de la repetición
        for frame in rep:

            landmarks = frame.reshape(33, 3)

            # Obtenemos las coordenadas de las caderas
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            # Obtenemos las coordenadas de las rodillas
            left_knee = landmarks[25]
            right_knee = landmarks[26]

            # Obtenemos las coordenadas de los tobillos
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]

            # Obtenemos las coordenadas de los hombros
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            # Calculamos el ángulo de la rodilla izquierda
            ba = left_hip - left_knee
            bc = left_ankle - left_knee

            norm = np.linalg.norm(ba) * np.linalg.norm(bc)

            if norm > 0:

                cosine = np.dot(ba, bc) / norm
                cosine = np.clip(cosine, -1, 1)

                left_angle = np.degrees(
                    np.arccos(cosine)
                )

                left_knee_angles.append(left_angle)

            # Calculamos el ángulo de la rodilla derecha
            ba = right_hip - right_knee
            bc = right_ankle - right_knee

            norm = np.linalg.norm(ba) * np.linalg.norm(bc)

            if norm > 0:

                cosine = np.dot(ba, bc) / norm
                cosine = np.clip(cosine, -1, 1)

                right_angle = np.degrees(
                    np.arccos(cosine)
                )

                right_knee_angles.append(right_angle)

            # Guardamos la posición horizontal de las rodillas
            left_knee_x.append(left_knee[0])
            right_knee_x.append(right_knee[0])

            # Guardamos la posición horizontal de los tobillos
            left_ankle_x.append(left_ankle[0])
            right_ankle_x.append(right_ankle[0])

            # Guardamos la posición vertical de las caderas
            left_hip_y.append(left_hip[1])
            right_hip_y.append(right_hip[1])

            # Guardamos la posición horizontal de las caderas
            left_hip_x.append(left_hip[0])
            right_hip_x.append(right_hip[0])

            # Calculamos el ángulo del tronco respecto a la vertical
            mid_shoulder = (
                left_shoulder +
                right_shoulder
            ) / 2

            mid_hip = (
                left_hip +
                right_hip
            ) / 2

            torso_vector = mid_shoulder - mid_hip

            torso_angle = abs(
                np.degrees(
                    np.arctan2(
                        torso_vector[0],
                        torso_vector[1] + 1e-6
                    )
                )
            )

            torso_angles.append(torso_angle)

        # Convertimos las variables a arrays
        left_knee_angles = np.array(left_knee_angles)
        right_knee_angles = np.array(right_knee_angles)

        left_knee_x = np.array(left_knee_x)
        right_knee_x = np.array(right_knee_x)

        left_ankle_x = np.array(left_ankle_x)
        right_ankle_x = np.array(right_ankle_x)

        left_hip_y = np.array(left_hip_y)
        right_hip_y = np.array(right_hip_y)

        left_hip_x = np.array(left_hip_x)
        right_hip_x = np.array(right_hip_x)

        torso_angles = np.array(torso_angles)

        # Comprobamos la simetría de las piernas
        if len(left_knee_angles) > 5 and len(right_knee_angles) > 5:

            knee_difference = np.mean(
                np.abs(
                    left_knee_angles -
                    right_knee_angles
                )
            )

            if knee_difference > 15:

                errors.append(
                    "Se observa una falta de simetría entre ambas piernas durante la sentadilla."
                )

        # Calculamos la profundidad de la sentadilla
        if len(left_hip_y) > 5 and len(right_hip_y) > 5:

            hip_y = (
                left_hip_y +
                right_hip_y
            ) / 2

            hip_movement = np.ptp(hip_y)

            # Comprobamos si existe suficiente desplazamiento vertical
            if hip_movement < 0.20:

                errors.append(
                    "No se alcanza una profundidad suficiente en la sentadilla."
                )

        # Calculamos la distancia entre cada rodilla y su tobillo correspondiente
        if (
            len(left_knee_x) > 5 and
            len(right_knee_x) > 5
        ):

            left_knee_ankle_diff = np.abs(
                left_knee_x -
                left_ankle_x
            )

            right_knee_ankle_diff = np.abs(
                right_knee_x -
                right_ankle_x
            )

            # Calculamos el desplazamiento máximo de las rodillas respecto a los tobillos
            left_lateral_displacement = np.max(
                left_knee_ankle_diff
            )

            right_lateral_displacement = np.max(
                right_knee_ankle_diff
            )

            # Comprobamos si existe un desplazamiento lateral excesivo de las rodillas
            if (
                left_lateral_displacement > 0.25
                or
                right_lateral_displacement > 0.25
            ):

                errors.append(
                    "Se observa un desplazamiento lateral excesivo de las rodillas respecto a los pies."
                )

        # Calculamos la variabilidad de la trayectoria de las rodillas
        if len(left_knee_x) > 5 and len(right_knee_x) > 5:

            left_knee_std = np.std(
                left_knee_x
            )

            right_knee_std = np.std(
                right_knee_x
            )

            # Comprobamos si las rodillas presentan una trayectoria inestable
            if (
                left_knee_std > 0.08
                or
                right_knee_std > 0.08
            ):

                errors.append(
                    "Las rodillas presentan una trayectoria poco estable durante la repetición."
                )

        # Comprobamos la estabilidad del tronco
        if len(torso_angles) > 5:

            torso_variation = np.ptp(
                torso_angles
            )

            # Comprobamos si existe una variación excesiva del tronco
            if torso_variation > 20:

                errors.append(
                    "Se observa una inclinación excesiva o poco estable del tronco durante la sentadilla."
                )

        # Calculamos la diferencia horizontal entre ambas caderas
        if len(left_hip_x) > 5 and len(right_hip_x) > 5:

            hip_difference = np.mean(
                np.abs(
                    left_hip_x -
                    right_hip_x
                )
            )

            # Comprobamos la simetría horizontal de las caderas
            if hip_difference > 0.15:

                errors.append(
                    "Se observa una asimetría en la posición de las caderas durante la ejecución."
                )

        # Mostramos los errores detectados
        if errors:

            print("Errores detectados:")

            for error in errors:

                print(f"- {error}")

        else:

            print(
                "No se ha podido determinar un error biomecánico específico."
            )


# Mostramos el resumen final
print("\nEvaluación:")

print(f"Repeticiones evaluadas: {len(repetitions)}")
print(f"Valid squat: {valid_count}")
print(f"Invalid squat: {invalid_count}")