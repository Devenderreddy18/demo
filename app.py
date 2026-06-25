"""
Real-Time Sign Language Recognition
Converted from Jupyter Notebook into a single Python script.
This version stores collected data in a separate dataset file and skips re-collection when data already exists.
"""

import json
import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pyttsx3

DATA_DIR = "custom_dataset"
DATA_INFO_FILE = "dataset_info.json"
DATA_FILE = "dataset_data.npz"
MODEL_FILE = "sign_model.h5"
LABELS_FILE = "labels.txt"
LABELS = ["hello", "i love you", "yes", "thanks", "no", "Like"]
NUM_SAMPLES = 300
MIN_CONFIDENCE = 0.8


def ensure_data_directories(labels, save_dir=DATA_DIR):
    for label in labels:
        os.makedirs(os.path.join(save_dir, label), exist_ok=True)


def save_dataset_info(labels, data_file=DATA_FILE, info_file=DATA_INFO_FILE):
    info = {
        "labels": labels,
        "data_file": data_file,
        "data_dir": DATA_DIR,
        "num_samples": NUM_SAMPLES,
    }
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)


def load_dataset_info(info_file=DATA_INFO_FILE):
    if not os.path.exists(info_file):
        return None
    with open(info_file) as f:
        return json.load(f)


def raw_data_ready(labels, save_dir=DATA_DIR, num_samples=NUM_SAMPLES):
    if not os.path.isdir(save_dir):
        return False
    for label in labels:
        folder = os.path.join(save_dir, label)
        if not os.path.isdir(folder):
            return False
        txt_files = [name for name in os.listdir(folder) if name.endswith(".txt")]
        if len(txt_files) < num_samples:
            return False
    return True


def collect_data(labels=LABELS, num_samples=NUM_SAMPLES, save_dir=DATA_DIR):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        cap = cv2.VideoCapture(0)
        ensure_data_directories(labels, save_dir)

        for label in labels:
            existing_files = sorted(
                [name for name in os.listdir(os.path.join(save_dir, label)) if name.endswith(".txt")]
            )
            collected = len(existing_files)
            print(f"Collecting data for '{label}' ({collected}/{num_samples} existing samples)")

            while collected < num_samples:
                ret, frame = cap.read()
                if not ret:
                    continue

                image = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0]
                    data = []
                    for lm in landmarks.landmark:
                        data.extend([lm.x, lm.y, lm.z])

                    file_path = os.path.join(save_dir, label, f"{label}_{collected}.txt")
                    with open(file_path, "w") as f:
                        f.write(",".join(map(str, data)))

                    collected += 1
                    cv2.putText(
                        image,
                        f"{label}: {collected}/{num_samples}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Collecting Data", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Data collection stopped by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            print(f"Done collecting {num_samples} samples for '{label}'.")

        cap.release()
        cv2.destroyAllWindows()



def build_dataset(labels=LABELS, save_dir=DATA_DIR, data_file=DATA_FILE):
    X, y = [], []

    for idx, label in enumerate(labels):
        folder = os.path.join(save_dir, label)
        for file in sorted(os.listdir(folder)):
            if not file.endswith(".txt"):
                continue
            points = np.loadtxt(os.path.join(folder, file), delimiter=",")
            X.append(points)
            y.append(idx)

    X = np.array(X)
    y = to_categorical(np.array(y))
    np.savez_compressed(data_file, X=X, y=y)
    print(f"Saved dataset file: {data_file}")
    return X, y


def load_data(data_file=DATA_FILE, labels=LABELS):
    if os.path.exists(data_file):
        data = np.load(data_file)
        print(f"Loaded dataset from {data_file}")
        return data["X"], data["y"]

    return build_dataset(labels=labels)


def save_labels(labels, label_file=LABELS_FILE):
    with open(label_file, "w") as f:
        f.write("\n".join(labels))


def load_labels(label_file=LABELS_FILE):
    if os.path.exists(label_file):
        with open(label_file) as f:
            return f.read().splitlines()
    return LABELS


def train_model(X, y, labels=LABELS):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(128, activation="relu", input_shape=(X.shape[1],)),
        Dense(64, activation="relu"),
        Dense(len(labels), activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    model.save(MODEL_FILE)
    save_labels(labels)
    print(f"Saved model to {MODEL_FILE}")
    return model


def recognize_signs(model, labels, min_confidence=MIN_CONFIDENCE):
    engine = pyttsx3.init()
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            image = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                data = []
                for lm in hand.landmark:
                    data.extend([lm.x, lm.y, lm.z])

                if len(data) == model.input_shape[1]:
                    prediction = model.predict(np.array([data]))[0]
                    class_id = np.argmax(prediction)
                    confidence = prediction[class_id]
                    if confidence > min_confidence:
                        label_text = f"{labels[class_id]}: {confidence * 100:.1f}%"
                        cv2.putText(
                            image,
                            label_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2,
                        )
                        engine.say(labels[class_id])
                        engine.runAndWait()

            cv2.imshow("Sign Recognition", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    dataset_info = load_dataset_info()
    if os.path.exists(DATA_FILE):
        print("Existing dataset file found. Skipping data collection.")
    elif raw_data_ready(LABELS):
        print("Raw data files found. Building dataset file.")
        build_dataset(labels=LABELS)
        save_dataset_info(LABELS)
    else:
        print("Collecting sign language data. Press 'q' to stop early.")
        collect_data(labels=LABELS)
        if raw_data_ready(LABELS):
            build_dataset(labels=LABELS)
            save_dataset_info(LABELS)
        else:
            raise RuntimeError("Data collection did not complete. Please rerun the script after collecting enough data.")

    X, y = load_data(data_file=DATA_FILE, labels=LABELS)
    model = train_model(X, y, labels=LABELS)
    recognize_signs(model, load_labels())


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)  