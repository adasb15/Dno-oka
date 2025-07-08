import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from sklearn.metrics import confusion_matrix
from utils.processing import generate_fov_mask_by_brightness, preprocess_image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt

MODEL_PATH = "unet_model.h5"  # Ścieżka zapisu modelu

# === Metryka IoU (Intersection over Union) ===
def iou_score(y_pred, y_true, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    y_true = y_true.float()

    intersection = (y_pred * y_true).sum((1, 2, 3))
    union = ((y_pred + y_true) >= 1).float().sum((1, 2, 3))
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.mean().item()

# === Własna funkcja straty: BCE + Jaccard Loss ===
class BCEJaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.jaccard = smp.losses.JaccardLoss(mode='binary')  # 1 - IoU

    def forward(self, y_pred, y_true):
        return self.bce(y_pred, y_true) + self.jaccard(y_pred, y_true)

# === Trenowanie modelu UNet ===
def generate_model(files, IMAGE_DIR, MASK_DIR):
    # Usunięcie 5 obrazów testowych
    files.remove("im0077.ppm")
    files.remove("im0081.ppm")
    files.remove("im0082.ppm")
    files.remove("im0162.ppm")
    files.remove("im0163.ppm")

    np.random.shuffle(files)

    train_files = files

    images = []
    masks = []

    # Wczytywanie obrazów i masek
    for fname in train_files:
        img_path = os.path.join(IMAGE_DIR, fname)
        mask_path = os.path.join(MASK_DIR, os.path.splitext(fname)[0] + ".ah.ppm")

        if not os.path.exists(mask_path):
            continue

        # Obraz RGB + normalizacja
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255

        # Maska ekspercka: binarna
        mask = Image.open(mask_path).convert("L")
        mask = (np.array(mask) > 127).astype(np.uint8)

        images.append(image)
        masks.append(mask)

    # Podział danych: train/val/test = 80% / 10% / 10%
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(images, masks, test_size=0.1, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=1/9, random_state=42)

    # Konwersja do tensora i transpozycja do formatu [B, C, H, W]
    def prepare_data(X, Y):
        X = np.array(X)
        Y = np.expand_dims(np.array(Y), axis=-1)
        X = torch.tensor(X).permute(0, 3, 1, 2).float()
        Y = torch.tensor(Y).permute(0, 3, 1, 2).float()
        return X, Y

    X_train, Y_train = prepare_data(X_train, Y_train)
    X_val, Y_val = prepare_data(X_val, Y_val)
    X_test, Y_test = prepare_data(X_test, Y_test)

    # Informacje diagnostyczne
    st.write("X_train shape:", X_train.shape)
    st.write("Y_train shape:", Y_train.shape)
    st.write("X_valid shape:", X_val.shape)
    st.write("Y_valid shape:", Y_val.shape)
    st.write("X_test shape:", X_test.shape)
    st.write("Y_test shape:", Y_test.shape)

    # === Definicja modelu UNet (z VGG16 jako encoder) ===
    model = smp.Unet(
        encoder_name='vgg16',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )

    loss_fn = BCEJaccardLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    best_model = model
    iter_worst = 0  # Licznik pogarszających się epok

    # === Trenowanie ===
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        preds = torch.sigmoid(preds)
        loss = loss_fn(preds, Y_train)
        loss.backward()
        optimizer.step()

        # Ewaluacja na zbiorze walidacyjnym
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = loss_fn(torch.sigmoid(val_preds), Y_val).item()

        st.write(f"Epoch {epoch+1}, Train loss: {loss.item():.4f}, Val loss: {val_loss:.4f}")

        # Early stopping: zapisz model tylko jeśli poprawia wynik
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            iter_worst = 0
        else:
            iter_worst += 1
            if iter_worst == 3:
                torch.save(best_model, MODEL_PATH)
                break

    # === Ewaluacja końcowa na train/val/test ===
    best_model.eval()
    with torch.no_grad():
        train_iou = iou_score((torch.sigmoid(best_model(X_train)) > 0.5).float(), Y_train)
        val_iou = iou_score((torch.sigmoid(best_model(X_val)) > 0.5).float(), Y_val)
        test_iou = iou_score((torch.sigmoid(best_model(X_test)) > 0.5).float(), Y_test)

    # === Wyświetlenie wyników IOU ===
    st.markdown(f"### 📊 Train IoU\n- **Train IoU**: `{train_iou:.4f}`")
    st.markdown(f"### 📊 Val IoU\n- **Val IoU**: `{val_iou:.4f}`")
    st.markdown(f"### 📊 Test IoU\n- **Test IoU**: `{test_iou:.4f}`")

    # Zapis najlepszego modelu
    torch.save(best_model, MODEL_PATH)
    st.success(f"Model zapisany do {MODEL_PATH}")

# === Przewidywanie maski z wytrenowanego modelu ===
def predict_mask_from_model(image_rgb, model):
    # Normalizacja obrazu i konwersja do tensora
    image = image_rgb.astype(np.float32) / 255.0
    tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()

    model.eval()
    with torch.no_grad():
        output = model(tensor)                      # Wynik surowy
        pred_mask = output.squeeze().cpu().numpy()  # Konwersja do numpy
        binary_mask = (pred_mask > 0.5).astype(np.uint8)  # Binarne progowanie
        return binary_mask
