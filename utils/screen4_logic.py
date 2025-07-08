import streamlit as st
import os
import numpy as np
import cv2
import joblib
from imblearn.pipeline import Pipeline
from skimage.measure import moments_central, moments_hu
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from PIL import Image

from utils.processing import preprocess_image, generate_fov_mask_by_brightness, postprocess_mask


# === Ekstrakcja cech z wycinka 5x5 ===
def extract_patch_features(patch):
    features = [np.mean(patch), np.var(patch)]  # Średnia i wariancja
    moments = moments_central(patch)            # Momenty centralne
    hu = moments_hu(moments)                    # Niezmienniki Hu (7 cech)
    features.extend(hu)
    return np.array(features)

# === Ekstrakcja etykiety klasy z patcha ===
def extract_labels(patch):
    # Jeśli środek patcha należy do naczynia (maski), oznacz jako 1
    return 1 if patch[2, 2] > 127 else 0

# === Przetwarzanie obrazu piksel po pikselu tylko wewnątrz FOV ===
def process_with_fov(h, w, fov, data, fun, output):
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            if fov[y, x]:
                patch = data[y - 2 : y + 3, x - 2 : x + 3]
                output.append(fun(patch))

# === Trenowanie klasyfikatora na wybranych obrazach ===
def generate_classifier(files, IMAGE_DIR, MASK_DIR):
    # Wyklucz obrazy testowe
    files.remove("im0077.ppm")
    files.remove("im0081.ppm")
    files.remove("im0082.ppm")
    files.remove("im0162.ppm")
    files.remove("im0163.ppm")

    np.random.shuffle(files)

    train_len = len(files)
    train_files = files[:train_len]

    images, masks, fov = [], [], []

    # Wczytaj obrazy, maski i maski FOV
    for fname in train_files:
        image_path = os.path.join(IMAGE_DIR, fname)
        mask_path = os.path.join(MASK_DIR, os.path.splitext(fname)[0] + ".ah.ppm")

        if not os.path.exists(mask_path):
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.array(image)

        preprocessed = preprocess_image(image_np)
        mask_fov = generate_fov_mask_by_brightness(image_np)

        expert = Image.open(mask_path).convert("L")
        expert_mask = (np.array(expert) > 127).astype(np.uint8)

        images.append(preprocessed)
        masks.append(expert_mask * 255)
        fov.append(mask_fov * 255)

    # Ekstrakcja cech i etykiet
    X, y = [], []
    for i in range(train_len):
        h, w = masks[i].shape
        process_with_fov(h, w, fov[i], masks[i], extract_labels, y)
        process_with_fov(h, w, fov[i], images[i], extract_patch_features, X)

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Statystyki przed undersamplingiem
    st.write("Train set size: ", len(X_train))
    st.write('X_train shape: ', np.array(X_train).shape)
    st.write('y_train shape: ', np.array(y_train).shape)
    st.write('y_train label 1: ', np.count_nonzero(y_train))
    st.write('y_train label 0: ', len(y_train) - np.count_nonzero(y_train))

    # Balansowanie danych
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    # Statystyki po undersamplingu
    st.write("Train set size after undersampling: ", len(X_train))
    st.write('y_train label 1: ', np.count_nonzero(y_train))
    st.write('y_train label 0: ', len(y_train) - np.count_nonzero(y_train))

    # Budowa modelu (Random Forest) i pipeline
    model = RandomForestClassifier()
    pipeline = Pipeline([
        ('rfc', model)
    ])

    # Siatka parametrów do GridSearchCV
    param_grid = {
        'rfc__n_estimators': [5, 150],
        'rfc__max_depth': [None, 30],
        'rfc__min_samples_leaf': [0, 3],
        'rfc__min_samples_split': [1, 10]
    }

    # Przeszukiwanie siatki hiperparametrów
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Wyniki przeszukiwania
    st.write("Najlepsze parametry: ", grid_search.best_params_)
    st.write("Najlepszy wynik: ", grid_search.best_score_)
    st.write('Training accuracy:', grid_search.score(X_train, y_train))
    st.write('Test accuracy:', grid_search.score(X_test, y_test))

    # Zapis modelu
    joblib.dump(grid_search, "model.joblib")

# === Wczytanie zapisanego modelu ===
def load_model(path):
    try:
        return joblib.load(path)
    except:
        return None

# === Generowanie maski naczyń przy użyciu wytrenowanego modelu ===
def predict_mask_from_model(image, model, fov_mask, patch_size=5):
    h, w = image.shape
    mask_pred = np.zeros((h, w), dtype=np.uint8)
    features = []
    coords = []

    # Ekstrakcja cech z patchy tylko wewnątrz FOV
    for y in range(patch_size // 2, h - patch_size // 2):
        for x in range(patch_size // 2, w - patch_size // 2):
            if fov_mask[y, x]:
                patch = image[y - 2:y + 3, x - 2:x + 3]
                features.append(extract_patch_features(patch))
                coords.append((y, x))

    if not features:
        return mask_pred  # Jeśli brak cech, zwróć pustą maskę

    # Przewidywanie klasy pikseli
    features = np.array(features)
    predictions = model.predict(features)

    # Wypełnij maskę na podstawie przewidywań
    for (y, x), pred in zip(coords, predictions):
        if pred == 1:
            mask_pred[y, x] = 1

    return mask_pred * fov_mask  # Uwzględnij tylko obszar FOV
