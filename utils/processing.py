import cv2
import numpy as np
import streamlit as st
from skimage.filters import frangi
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.metrics import sensitivity_score, specificity_score

# === Tworzenie maski FOV na podstawie jasności ===
def generate_fov_mask_by_brightness(image, threshold=40):
    # Konwersja do skali szarości, jeśli obraz jest kolorowy
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Utworzenie maski na podstawie progu jasności
    mask = (gray > threshold)
    return mask

# === Wstępne przetwarzanie obrazu (kanał zielony) ===
def preprocess_image(image):
    # Wydzielenie kanału zielonego
    green_channel = image[:, :, 1]

    # Obcięcie wartości skrajnych
    clipped = np.clip(green_channel, 10, 245)

    # Rozmycie Gaussa w celu redukcji szumu
    blurred = cv2.GaussianBlur(clipped, (9, 9), 0)

    # Wyostrzenie obrazu przez maskowanie nieostrości
    sharpened = cv2.addWeighted(blurred, 1.5, blurred, -0.5, 0)

    # Lokalna normalizacja histogramu (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)

    return enhanced

# === Filtr Frangiego do wykrywania naczyń krwionośnych ===
def apply_frangi_filter(image, fov_mask=None):
    # Zastosowanie filtru Frangiego
    filtered = frangi(image, sigmas=np.arange(1, 5, 0.5), black_ridges=True)

    # Normalizacja do zakresu [0, 1]
    filtered = filtered / filtered.max()

    # Ograniczenie do obszaru FOV (jeśli podano)
    if fov_mask is not None:
        filtered = filtered * fov_mask

    # Progowanie – konwersja do maski binarnej
    binary_mask = np.where(filtered > 0.03, 1, 0)
    return binary_mask

# === Czyszczenie maski binarnej – operacje morfologiczne ===
def postprocess_mask(binary_mask):
    # Konwersja do typu uint8, jeśli trzeba
    if binary_mask.dtype != np.uint8:
        binary_mask = (binary_mask * 255).astype(np.uint8) if binary_mask.max() <= 1 else binary_mask.astype(np.uint8)

    # Otwarcie morfologiczne w celu usunięcia szumu
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return opened / 255  # Zwróć maskę w postaci float (0–1)

# === Ewaluacja maski względem maski eksperckiej ===
def evaluate_mask(y_true, y_pred):
    # Spłaszczanie do 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Obliczanie metryk klasyfikacyjnych
    acc = accuracy_score(y_true, y_pred)
    sens = sensitivity_score(y_true, y_pred, pos_label=1)
    spec = specificity_score(y_true, y_pred, pos_label=1)
    mean_arith = (sens + spec) / 2
    mean_geom = np.sqrt(sens * spec)
    conf = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "mean_arithmetic": mean_arith,
        "mean_geometric": mean_geom,
        "confusion_matrix": conf
    }

# === Nakładanie maski binarnej na obraz RGB (do wizualizacji) ===
def visualize_overlay(original_image, binary_mask):
    overlay = original_image.copy()

    # Upewnij się, że maska jest binarna (0/1)
    mask = (binary_mask > 0).astype(np.uint8)

    # Dopasuj rozmiar obrazu, jeśli jest inny niż maski
    if mask.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (mask.shape[1], mask.shape[0]))

    # Zastąp piksele w masce kolorem czerwonym [255, 0, 0]
    overlay[mask == 1] = [255, 0, 0]

    return overlay
