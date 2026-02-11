import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset"
IMG_SIZE = 128
SEQUENCE_LENGTH = 30
BATCH_SIZE = 4
EPOCHS = 10

def load_dataset():
    sequences = []
    labels = []
    class_names = os.listdir(DATASET_PATH)

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(DATASET_PATH, class_name)

        for video_name in tqdm(os.listdir(class_path), desc=f"Loading {class_name}"):

            video_path = os.path.join(class_path, video_name)
            cap = cv2.VideoCapture(video_path)

            frames = []

            while len(frames) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = frame / 255.0
                frames.append(frame)

            cap.release()

            if len(frames) == SEQUENCE_LENGTH:
                sequences.append(frames)
                labels.append(label)

    return np.array(sequences), np.array(labels), class_names


print("Loading dataset...")

if not os.path.exists(DATASET_PATH):
    print("❌ dataset folder not found!")
    exit()

X, y, class_names = load_dataset()

if len(X) == 0:
    print("❌ No video data found inside dataset folders!")
    exit()

y = tf.keras.utils.to_categorical(y, num_classes=len(class_names))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset shape:", X_train.shape)

model = tf.keras.Sequential([
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)
    ),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2,2)),

    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(64, (3,3), activation='relu')
    ),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2,2)),

    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),

    tf.keras.layers.LSTM(128),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

model.save("large_isl_model.h5")

print("✅ Training complete. Model saved as large_isl_model.h5")

