import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Función que normaliza un frame a formato (33,3)
def normalize_frame(frame):

    frame = np.array(frame)

    try:

        # caso 1: (132,)
        # el vector de 132 elementos
        # 33 landmarks * 4 coordenadas (x,y,z,visibility)
        if frame.shape == (132,):
            frame = frame.reshape(33, 4)
            frame = frame[:, :3]  # quitar visibility
            return frame

        # caso 2: (1,132)
        # a veces el frame viene con una dimensión extra al principio
        # (1,132) -> (132,) -> (33,4) -> (33,3)
        # en ese caso, primero quitar la dimensión extra y luego procesar como el caso 1
        if len(frame.shape) == 3 and frame.shape[0] == 1:
            frame = frame[0]
            if frame.shape == (132,):
                frame = frame.reshape(33, 4)
                frame = frame[:, :3]  # quitar visibility
                return frame

        # caso 3: (33,4)
        # a veces el frame ya viene con formato (33,4)
        # en ese caso, simplemente quitar la columna de visibility
        if frame.shape == (33, 4):
            return frame[:, :3]

        # caso 4: (33,3)
        # a veces el frame ya viene con formato (33,3)
        # en ese caso, no hay que hacer nada
        if frame.shape == (33, 3):
            return frame

        return None

    except:
        return None


# Función que carga un video a partir de una carpeta con frames en formato .npy
def load_video(video_path):

    frames = []

    files = sorted([
        f for f in os.listdir(video_path)
        if f.endswith(".npy")
    ])

    for f in files:

        try:
            frame = np.load(os.path.join(video_path, f), allow_pickle=True)

            frame = normalize_frame(frame)

            if frame is None:
                continue

            frames.append(frame)

        except:
            continue

    return np.array(frames)


# Función que carga el dataset a partir de la estructura de carpetas
def load_dataset(base_path):

    # X: lista de videos (cada video es un array de frames, cada frame es un array de 33x3)
    # y: lista de etiquetas (1 para Valid, 0 para Invalid)
    X, y = [], []

    for label_name in ["Valid", "Invalid"]:

        class_folder = os.path.join(base_path, label_name)

        if not os.path.exists(class_folder):
            print(f"No existe: {class_folder}")
            continue

        print(f"\nCargando clase: {label_name}")

        for video_folder in os.listdir(class_folder):

            video_path = os.path.join(class_folder, video_folder)

            if not os.path.isdir(video_path):
                continue

            video = load_video(video_path)


            # solo cargar videos con al menos 5 frames
            if len(video) < 5:
                continue

            X.append(video)
            y.append(1 if label_name == "Valid" else 0)

    print("\nTotal videos cargados:", len(X))

    return X, np.array(y)


# Función que establece una longitud fija para cada video, recortando o rellenando con ceros según sea necesario
def pad_sequences(X, max_len=60):

    X_pad = []

    for seq in X:

        # (frames, 33, 3) -> (frames, 99)
        seq = seq.reshape(len(seq), -1)

        if len(seq) > max_len:
            seq = seq[:max_len]

        else:
            pad = np.zeros((max_len - len(seq), 99))
            seq = np.concatenate([seq, pad], axis=0)

        X_pad.append(seq)

    return np.array(X_pad, dtype=np.float32)


# pipeline
X, y = load_dataset("datasets")

if len(X) == 0:
    raise ValueError("No se han cargado datos.")

X = pad_sequences(X, max_len=60)

print("\nShape X:", X.shape)
print("Shape y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Sequential([
    LSTM(
        128,
        return_sequences=True,
        input_shape=(60, 99)
    ),
    Dropout(0.3),

    LSTM(64),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32
)

# evaluación del modelo
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print("\nResultados en el conjunto de prueba:")
print("Loss:", loss)
print("Accuracy:", accuracy)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Invalid", "Valid"]))

cm = confusion_matrix(y_test, y_pred)

print("\nMatriz de Confusión:")
print(cm)

model.save("squat_model.h5")