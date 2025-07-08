import numpy as np
import streamlit as st
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
import pandas as pd

from utils.screen4_logic import load_model, predict_mask_from_model as predict_rf
from utils.screen5_logic import predict_mask_from_model as predict_unet
from utils.processing import *

IMAGE_DIR = "stare-images"
MASK_DIR = "labels-ah"
MODEL_PATH_RF = "model.joblib"
MODEL_PATH_UNET = "unet_model.h5"

SELECTED_FILES = ["im0077.ppm", "im0081.ppm", "im0082.ppm", "im0162.ppm", "im0163.ppm"]

def run_compare():
    st.header("🔀 Porównanie trzech metod segmentacji")

    model_rf = load_model(MODEL_PATH_RF)
    model_unet = torch.load(MODEL_PATH_UNET, weights_only=False)
    model_unet.eval()

    files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith((".ppm", ".png", ".jpg"))])

    # ======= Wybór obrazu i predykcja modelem =======
    st.subheader("📄 Predykcja dla wybranego obrazu na podstawie modelu")
    selected_file = st.selectbox("Wybierz obraz:", files, index=None)


    if selected_file:
        image_path = os.path.join(IMAGE_DIR, selected_file)
        mask_path = os.path.join(MASK_DIR, os.path.splitext(selected_file)[0] + ".ah.ppm")

        expert = Image.open(mask_path).convert("L")
        expert_mask = (np.array(expert) > 127).astype(np.uint8)

        # Wczytaj obraz
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.array(image)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Wyniki przetwarzania wybranego obrazu filtrem Frangi")
            with st.spinner("Przetwarzanie..."):
                preprocessed = preprocess_image(image_np)

                mask_fov = generate_fov_mask_by_brightness(image_np)

                vessel_mask = apply_frangi_filter(preprocessed / 255, mask_fov)

            overlay = visualize_overlay(image_np, vessel_mask)

            st.markdown("<div style='min-height:100px'><h3>Oryginalny obraz</h3></div>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)

            st.markdown("<div style='min-height:100px'><h3>Maska naczyń (binarna)</h3></div>", unsafe_allow_html=True)
            st.image(vessel_mask * 255, clamp=True, use_container_width=True)

            st.markdown("<div style='min-height:100px'><h3>Maska ekspercka</h3></div>", unsafe_allow_html=True)
            st.image(expert_mask * 255, clamp=True, use_container_width=True)

            st.markdown("<div style='min-height:100px'><h3>Obraz z nałożoną maską</h3></div>",
                        unsafe_allow_html=True)
            st.image(overlay, use_container_width=True)

            st.subheader("Miary porównawcze z maską ekspercką")
            results = evaluate_mask(expert_mask, vessel_mask)

            st.markdown(f"""
                            **Accuracy**: {results['accuracy']:.4f}  
                            **Sensitivity (Recall)**: {results['sensitivity']:.4f}  
                            **Specificity**: {results['specificity']:.4f}  
                            **Średnia arytmetyczna (Se + Sp / 2)**: {results['mean_arithmetic']:.4f}  
                            **Średnia geometryczna (sqrt(Se × Sp))**: {results['mean_geometric']:.4f}  
                            """)

            conf_matrix = results["confusion_matrix"]
            conf_df = pd.DataFrame(
                conf_matrix,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"]
            )
            st.markdown("**Macierz pomyłek**")
            st.dataframe(conf_df.style.format("{:.0f}"))

        with col2:
            st.subheader("Predykcja dla wybranego obrazu na podstawie klasyfikatora")
            with st.spinner("Przetwarzanie..."):
                fov_mask = generate_fov_mask_by_brightness(image_np)
                pred_mask = predict_rf(preprocessed, model_rf, fov_mask)
                overlay = visualize_overlay(image_np, pred_mask)

            st.markdown("<div style='min-height:100px'><h3>Oryginalny obraz</h3></div>", unsafe_allow_html=True)
            st.image(image_np, use_container_width=True)
            st.markdown("<div style='min-height:100px'><h3>Maska naczyń (binarna)</h3></div>",
                        unsafe_allow_html=True)
            st.image(pred_mask * 255, clamp=True, use_container_width=True)
            st.markdown("<div style='min-height:100px'><h3>Maska ekspercka</h3></div>", unsafe_allow_html=True)
            st.image(expert_mask * 255, clamp=True, use_container_width=True)
            st.markdown("<div style='min-height:100px'><h3>Obraz z nałożoną maską</h3></div>", unsafe_allow_html=True)
            st.image(overlay, clamp=True, use_container_width=True)

            st.subheader("Miary porównawcze z maską ekspercką")
            results = evaluate_mask(expert_mask, pred_mask)

            st.markdown(f"""
                            **Accuracy**: {results['accuracy']:.4f}  
                            **Sensitivity (Recall)**: {results['sensitivity']:.4f}  
                            **Specificity**: {results['specificity']:.4f}  
                            **Średnia arytmetyczna (Se + Sp / 2)**: {results['mean_arithmetic']:.4f}  
                            **Średnia geometryczna (sqrt(Se × Sp))**: {results['mean_geometric']:.4f}  
                            """)

            conf_matrix = results["confusion_matrix"]
            conf_df = pd.DataFrame(
                conf_matrix,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"]
            )
            st.markdown("**Macierz pomyłek**")
            st.dataframe(conf_df.style.format("{:.0f}"))

        with col3:
            st.subheader("Predykcja dla wybranego obrazu na podstawie modelu UNet")

            with st.spinner("Przetwarzanie..."):
                pred_mask = predict_unet(image, model_unet)
                pred_mask_copy = pred_mask.copy()
                pred_mask_copy[~fov_mask.astype(bool)] = 0
                overlay = visualize_overlay(image, pred_mask_copy)

            st.markdown("<div style='min-height:100px'><h3>Oryginalny obraz</h3></div>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("<div style='min-height:100px'><h3>Maska naczyń (binarna)</h3></div>",
                        unsafe_allow_html=True)
            st.image(pred_mask_copy * 255, clamp=True, use_container_width=True)
            st.markdown("<div style='min-height:100px'><h3>Maska ekspercka</h3></div>", unsafe_allow_html=True)
            st.image(expert_mask * 255, clamp=True, use_container_width=True)
            st.markdown("<div style='min-height:100px'><h3>Obraz z nałożoną maską</h3></div>", unsafe_allow_html=True)
            st.image(overlay, clamp=True, use_container_width=True)

            st.subheader("Miary porównawcze z maską ekspercką")
            results = evaluate_mask(expert_mask, pred_mask_copy)

            st.markdown(f"""
                            **Accuracy**: {results['accuracy']:.4f}  
                            **Sensitivity (Recall)**: {results['sensitivity']:.4f}  
                            **Specificity**: {results['specificity']:.4f}  
                            **Średnia arytmetyczna (Se + Sp / 2)**: {results['mean_arithmetic']:.4f}  
                            **Średnia geometryczna (sqrt(Se × Sp))**: {results['mean_geometric']:.4f}  
                            """)

            conf_matrix = results["confusion_matrix"]
            conf_df = pd.DataFrame(
                conf_matrix,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"]
            )
            st.markdown("**Macierz pomyłek**")
            st.dataframe(conf_df.style.format("{:.0f}"))

    analys_for_5_img = st.checkbox("Wykonaj analize na 5 wybranych obrazach", value=False)

    if analys_for_5_img:
        frangi_list = [[],[],[],[],[]]
        frangi_conf_matrix_sum = np.zeros((2, 2), dtype=int)

        rf_list = [[],[],[],[],[]]
        rf_conf_matrix_sum = np.zeros((2, 2), dtype=int)

        unet_list = [[],[],[],[],[]]
        unet_conf_matrix_sum = np.zeros((2, 2), dtype=int)

        for fname in SELECTED_FILES:
            st.subheader(f"Obraz: {fname}")

            img_path = os.path.join(IMAGE_DIR, fname)
            mask_path = os.path.join(MASK_DIR, os.path.splitext(fname)[0] + ".ah.ppm")

            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            np_image = np.array(image_rgb)
            preprocessed = preprocess_image(np_image)

            # === Maska ekspercka
            expert = Image.open(mask_path).convert("L")
            expert_mask = (np.array(expert) > 127).astype(np.uint8)
            expert_overlay = visualize_overlay(np_image, expert_mask)


            # === FOV
            fov_mask = generate_fov_mask_by_brightness(np_image)

            # === FRANGI
            with st.spinner("Przetwarzanie..."):
                frangi_mask = apply_frangi_filter(preprocessed / 255, fov_mask)
                frangi_mask = postprocess_mask(frangi_mask * 255)
                frangi_overlay = visualize_overlay(np_image, frangi_mask)
                results_frangi = evaluate_mask(expert_mask, frangi_mask)
                frangi_list[0].append(results_frangi['accuracy'])
                frangi_list[1].append(results_frangi['sensitivity'])
                frangi_list[2].append(results_frangi['specificity'])
                frangi_list[3].append(results_frangi['mean_arithmetic'])
                frangi_list[4].append(results_frangi['mean_geometric'])

                # === RANDOM FOREST (screen4)
                rf_mask = predict_rf(preprocessed, model_rf, fov_mask)
                rf_overlay = visualize_overlay(np_image, rf_mask)
                results_rf = evaluate_mask(expert_mask, rf_mask)
                rf_list[0].append(results_rf['accuracy'])
                rf_list[1].append(results_rf['sensitivity'])
                rf_list[2].append(results_rf['specificity'])
                rf_list[3].append(results_rf['mean_arithmetic'])
                rf_list[4].append(results_rf['mean_geometric'])

                # === UNET (screen5)
                unet_mask = predict_unet(np_image, model_unet)
                unet_overlay = visualize_overlay(np_image, unet_mask)
                results_unet = evaluate_mask(expert_mask, unet_mask)
                unet_list[0].append(results_unet['accuracy'])
                unet_list[1].append(results_unet['sensitivity'])
                unet_list[2].append(results_unet['specificity'])
                unet_list[3].append(results_unet['mean_arithmetic'])
                unet_list[4].append(results_unet['mean_geometric'])

            # === Porównawcza wizualizacja
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.image(expert_mask * 255, caption="Maska ekspercka", use_container_width=True)
                st.image(expert_overlay, clamp=True, caption="Nałożona maska ekspercka", use_container_width=True)
            with col2:
                st.image(frangi_mask * 255, clamp=True, caption="Frangi", use_container_width=True)
                st.image(frangi_overlay, clamp=True, caption="Nałożona maska Frangi", use_container_width=True)
            with col3:
                st.image(rf_mask * 255, clamp=True, caption="Random Forest", use_container_width=True)
                st.image(rf_overlay, clamp=True, caption="Nałożona maska Random Forest", use_container_width=True)
            with col4:
                st.image(unet_mask * 255, clamp=True, caption="UNet", use_container_width=True)
                st.image(unet_overlay, clamp=True, caption="Nałożona maska UNet", use_container_width=True)

            # === Wyniki
            df = pd.DataFrame({
                "Metoda": ["Frangi", "Random Forest", "UNet"],
                "Accuracy": [
                    results_frangi["accuracy"],
                    results_rf["accuracy"],
                    results_unet["accuracy"]
                ],
                "Sensitivity": [
                    results_frangi["sensitivity"],
                    results_rf["sensitivity"],
                    results_unet["sensitivity"]
                ],
                "Specificity": [
                    results_frangi["specificity"],
                    results_rf["specificity"],
                    results_unet["specificity"]
                ],
                "Mean Arithmetic": [
                    results_frangi["mean_arithmetic"],
                    results_rf["mean_arithmetic"],
                    results_unet["mean_arithmetic"]
                ],
                "Mean Geometric": [
                    results_frangi["mean_geometric"],
                    results_rf["mean_geometric"],
                    results_unet["mean_geometric"]
                ]
            })

            st.dataframe(df.style.format({ col:"{:.4f}" for col in df.columns if col != "Metoda"}))

            # === Macierze pomyłek
            st.markdown("### 🔍 Macierze pomyłek")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Frangi**")
                conf_fra = pd.DataFrame(
                    results_frangi["confusion_matrix"],
                    index=["AN", "AP"],
                    columns=["PN", "PP"]
                )
                frangi_conf_matrix_sum += results_frangi["confusion_matrix"]

                st.dataframe(conf_fra.style.format("{:.0f}"))

            with col2:
                st.markdown("**Random Forest**")
                conf_rf = pd.DataFrame(
                    results_rf["confusion_matrix"],
                    index=["AN", "AP"],
                    columns=["PN", "PP"]
                )
                rf_conf_matrix_sum += results_rf["confusion_matrix"]

                st.dataframe(conf_rf.style.format("{:.0f}"))

            with col3:
                st.markdown("**UNet**")
                conf_unet = pd.DataFrame(
                    results_unet["confusion_matrix"],
                    index=["AN", "AP"],
                    columns=["PN", "PP"]
                )
                unet_conf_matrix_sum += results_unet["confusion_matrix"]

                st.dataframe(conf_unet.style.format("{:.0f}"))

        st.write("Legenda:")
        st.write("AN - Actual Negative")
        st.write("AP - Actual Positive")
        st.write("PN - Predicted Negative")
        st.write("PP - Predicted Positive")

        with col1:
            st.subheader("Podsumowanie zbiorcze wyników dla Frangi:")

            st.markdown(f"""
                            - **Średnia accuracy**: {np.mean(frangi_list[0]):.4f}  
                            - **Średnia sensitivity (Recall)**: {np.mean(frangi_list[1]):.4f}  
                            - **Średnia specificity**: {np.mean(frangi_list[2]):.4f}  
                            - **Średnia arytmetyczna (Se + Sp / 2)**: {np.mean(frangi_list[3]):.4f}  
                            - **Średnia geometryczna (sqrt(Se × Sp))**: {np.mean(frangi_list[4]):.4f}
                            """)

            df_summary = pd.DataFrame({
                "Obraz": SELECTED_FILES,
                "Accuracy": frangi_list[0],
                "Sensitivity": frangi_list[1],
                "Specificity": frangi_list[2],
                "Średnia Arytmetyczna": frangi_list[3],
                "Średnia Geometryczna": frangi_list[4]
            })
            st.dataframe(df_summary.style.format({col: "{:.4f}" for col in df_summary.columns if col != "Obraz"}))


            st.markdown("### Sumaryczna macierz pomyłek dla Frangi:")
            conf_sum_df = pd.DataFrame(
                frangi_conf_matrix_sum,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"]
            )
            st.dataframe(conf_sum_df.style.format("{:.0f}"))

        with col2:
            st.subheader("Podsumowanie zbiorcze wyników dla Random Forest:")

            st.markdown(f"""
                               - **Średnia accuracy**: {np.mean(rf_list[0]):.4f}  
                               - **Średnia sensitivity (Recall)**: {np.mean(rf_list[1]):.4f}  
                               - **Średnia specificity**: {np.mean(rf_list[2]):.4f}  
                               - **Średnia arytmetyczna (Se + Sp / 2)**: {np.mean(rf_list[3]):.4f}  
                               - **Średnia geometryczna (sqrt(Se × Sp))**: {np.mean(rf_list[4]):.4f}
                               """)

            # Możesz też wyświetlić tabelę porównawczą
            df_summary = pd.DataFrame({
                "Obraz": SELECTED_FILES,
                "Accuracy": rf_list[0],
                "Sensitivity": rf_list[1],
                "Specificity": rf_list[2],
                "Średnia Arytmetyczna": rf_list[3],
                "Średnia Geometryczna": rf_list[4]
            })
            st.dataframe(df_summary.style.format({col: "{:.4f}" for col in df_summary.columns if col != "Obraz"}))

            st.markdown("### Sumaryczna macierz pomyłek dla Random Forest:")
            conf_sum_df = pd.DataFrame(
                rf_conf_matrix_sum,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"]
            )
            st.dataframe(conf_sum_df.style.format("{:.0f}"))

        with col3:
            st.subheader("Podsumowanie zbiorcze wyników dla UNet:")

            st.markdown(f"""
                               - **Średnia accuracy**: {np.mean(unet_list[0]):.4f}  
                               - **Średnia sensitivity (Recall)**: {np.mean(unet_list[1]):.4f}  
                               - **Średnia specificity**: {np.mean(unet_list[2]):.4f}  
                               - **Średnia arytmetyczna (Se + Sp / 2)**: {np.mean(unet_list[3]):.4f}  
                               - **Średnia geometryczna (sqrt(Se × Sp))**: {np.mean(unet_list[4]):.4f}
                               """)

            # Możesz też wyświetlić tabelę porównawczą
            df_summary = pd.DataFrame({
                "Obraz": SELECTED_FILES,
                "Accuracy": unet_list[0],
                "Sensitivity": unet_list[1],
                "Specificity": unet_list[2],
                "Średnia Arytmetyczna": unet_list[3],
                "Średnia Geometryczna": unet_list[4]
            })
            st.dataframe(df_summary.style.format({col: "{:.4f}" for col in df_summary.columns if col != "Obraz"}))

            st.markdown("### Sumaryczna macierz pomyłek dla UNet:")
            conf_sum_df = pd.DataFrame(
                unet_conf_matrix_sum,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"]
            )
            st.dataframe(conf_sum_df.style.format("{:.0f}"))