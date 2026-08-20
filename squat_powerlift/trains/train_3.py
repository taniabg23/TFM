import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Masking, GRU
from tensorflow.keras.models import Sequential

MAX_LEN = 60
PAD_VALUE = -999.0


def save_history_csv(history, filepath="./csvs/training_history_3.csv"):

    keys = list(history.history.keys())
    epochs = len(history.history[keys[0]])

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)

        for i in range(epochs):
            row = [i + 1] + [history.history[k][i] for k in keys]
            writer.writerow(row)


def save_learning_curves(history, filepath="./csvs/learning_curves_3.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("No se pudo generar learning_curves_3.png (matplotlib no disponible).")
        return

    acc_key = "accuracy" if "accuracy" in history.history else None
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train_loss")
    axes[0].plot(history.history["val_loss"], label="val_loss")
    axes[0].set_title("Loss por epoca")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    if acc_key and val_acc_key:
        axes[1].plot(history.history[acc_key], label="train_accuracy")
        axes[1].plot(history.history[val_acc_key], label="val_accuracy")
        axes[1].set_title("Accuracy por epoca")
        axes[1].set_xlabel("Epoca")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.2)
    else:
        axes[1].text(0.5, 0.5, "Accuracy no disponible en history", ha="center", va="center")
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close(fig)

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


def pad_sequences(X, max_len=60, pad_value=-999.0):

    X_out = []

    for seq in X:

        # (frames, 33, 3) -> (frames, 99)
        seq = seq[:max_len].reshape(min(len(seq), max_len), -1).astype(np.float32)

        if len(seq) < max_len:
            pad = np.full((max_len - len(seq), 99), pad_value, dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        X_out.append(seq)

    return np.array(X_out, dtype=np.float32)


# pipeline
X, y = load_dataset("../datasets")

if len(X) == 0:
    raise ValueError("No se han cargado datos.")

X_train_val_raw, X_test_raw, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_val_raw, y_train_val,
    test_size=0.2,
    random_state=42,
    stratify=y_train_val
)

# Solo hacer padding, sin estandarización manual
X_train = pad_sequences(X_train_raw, max_len=MAX_LEN, pad_value=PAD_VALUE)
X_val = pad_sequences(X_val_raw, max_len=MAX_LEN, pad_value=PAD_VALUE)
X_test = pad_sequences(X_test_raw, max_len=MAX_LEN, pad_value=PAD_VALUE)

print("\nShape X_train:", X_train.shape)
print("Shape X_val:", X_val.shape)
print("Shape X_test:", X_test.shape)
print("Shape y:", y.shape)

model = Sequential([
    Masking(mask_value=PAD_VALUE, input_shape=(MAX_LEN, 99)),

    GRU(128, return_sequences=True),
    Dropout(0.3),

    GRU(64),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        "./models/bests/best_squat_model_3.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    callbacks=callbacks
)

save_history_csv(history, filepath="./csvs/training_history_3.csv")
save_learning_curves(history, filepath="./csvs/learning_curves_3.png")

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

model.save("./models/squat_model_final_3.h5")
model.save("./models/squat_model_final_3.keras")