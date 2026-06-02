import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from src.emotion_model import build_model
from config import IMG_SIZE, EMOTIONS

# Paths
TRAIN_DIR = "C:/Users/chand/Downloads/dataset/fer2013/train"
TEST_DIR =  "C:/Users/chand/Downloads/dataset/fer2013/test"
MODEL_SAVE_PATH = "models/emotion_model.h5"


# 📦 Load Dataset Function
def load_data(data_dir):
    data = []
    labels = []

    print(f"[INFO] Loading data from: {data_dir}")

    for emotion in EMOTIONS:
        path = os.path.join(data_dir, emotion)
        label = EMOTIONS.index(emotion)

        if not os.path.exists(path):
            print(f"[WARNING] Folder not found: {path}")
            continue

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            try:
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    continue

                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(label)

            except Exception as e:
                print(f"[ERROR] Skipping file: {img_path}")
                continue

    data = np.array(data, dtype="float32") / 255.0
    data = np.reshape(data, (-1, IMG_SIZE, IMG_SIZE, 1))

    labels = to_categorical(labels, num_classes=len(EMOTIONS))

    print(f"[INFO] Loaded {len(data)} samples from {data_dir}")

    return data, labels


# 📊 Load Train & Test Data
X_train, y_train = load_data(TRAIN_DIR)
X_test, y_test = load_data(TEST_DIR)


# 🧠 Build Model
model = build_model()
model.summary()


# 💾 Save Best Model Automatically
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)


# 🚀 Train Model
print("[INFO] Training started...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=64,
    callbacks=[checkpoint]
)


# 🧪 Evaluate Model
print("\n[INFO] Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test)

print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"❌ Test Loss: {loss:.4f}")


# 💾 Save Final Model (Optional)
model.save(MODEL_SAVE_PATH)
print(f"[INFO] Model saved at: {MODEL_SAVE_PATH}")