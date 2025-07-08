import pandas as pd

from utils.screen4_logic import *
from utils.processing import evaluate_mask, visualize_overlay, preprocess_image


def run_screen4():
    st.header("🔸 Klasyfikacja cech (Random Forest, patch 5x5)")

    IMAGE_DIR = "stare-images"
    MASK_DIR = "labels-ah"
    MODEL_PATH = "model.joblib"

    files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith((".ppm", ".png", ".jpg"))])

    # ======= Wybór obrazu i predykcja modelem =======
    st.subheader("📄 Predykcja dla wybranego obrazu na podstawie modelu")
    selected_file = st.selectbox("Wybierz obraz:", files, index=None)

    if selected_file:
        image_path = os.path.join(IMAGE_DIR, selected_file)
        mask_path = os.path.join(MASK_DIR, os.path.splitext(selected_file)[0] + ".ah.ppm")

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        np_image_rgb = np.array(image_rgb)

        preprocessed = preprocess_image(np_image_rgb)

        model = load_model(MODEL_PATH)
        if model is not None:
            fov_mask = generate_fov_mask_by_brightness(np_image_rgb)
            pred_mask = predict_mask_from_model(preprocessed, model, fov_mask)

            overlay = visualize_overlay(np_image_rgb, pred_mask)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("<div style='min-height:100px'><h3>Oryginalny obraz</h3></div>", unsafe_allow_html=True)
                st.image(np_image_rgb, use_container_width=True)
            with col2:
                st.markdown("<div style='min-height:100px'><h3>Maska naczyń (binarna)</h3></div>",
                            unsafe_allow_html=True)
                st.image(pred_mask * 255, clamp=True, use_container_width=True)

            expert = Image.open(mask_path).convert("L")
            expert_mask = (np.array(expert) > 127).astype(np.uint8)

            with col3:
                st.markdown("<div style='min-height:100px'><h3>Maska ekspercka</h3></div>", unsafe_allow_html=True)
                st.image(expert_mask * 255, clamp=True, use_container_width=True)
            with col4:
                st.markdown("<div style='min-height:100px'><h3>Obraz z nałożoną maską</h3></div>", unsafe_allow_html=True)
                st.image(overlay, clamp=True, use_container_width=True)

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
        else:
            st.info("Brak wytrenowanego modelu — najpierw go wygeneruj poniżej.")

    test_for_5_img = st.checkbox("Wykonaj analize na 5 obrazach", value=False)

    if test_for_5_img:
        st.subheader("Wyniki dla 5 obrazów")

        columns = st.columns(5)
        file = ["im0077.ppm", "im0081.ppm", "im0082.ppm", "im0162.ppm", "im0163.ppm"]

        # Przygotowanie list do zbierania wyników
        acc_list = []
        sens_list = []
        spec_list = []
        arith_list = []
        geom_list = []
        conf_matrix_sum = np.zeros((2, 2), dtype=int)

        for i in range(5):
            with columns[i]:
                selected_file = file[i]
                image_path = os.path.join(IMAGE_DIR, selected_file)
                mask_path = os.path.join(MASK_DIR, os.path.splitext(selected_file)[0] + ".ah.ppm")

                # Wczytaj obraz
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                np_image_rgb = np.array(image_rgb)

                with st.spinner("Przetwarzanie..."):
                    preprocessed = preprocess_image(np_image_rgb)
                    model = load_model(MODEL_PATH)

                    if model is not None:
                        fov_mask = generate_fov_mask_by_brightness(np_image_rgb)
                        pred_mask = predict_mask_from_model(preprocessed, model, fov_mask)

                        overlay = visualize_overlay(np_image_rgb, pred_mask)

                st.markdown("### Oryginalny obraz")
                st.image(np_image_rgb, use_container_width=True)

                st.markdown("### Maska naczyń (binarna)")
                st.image(pred_mask * 255, clamp=True, use_container_width=True)


                # === EWALUACJA ===
                if os.path.exists(mask_path):
                    expert = Image.open(mask_path).convert("L")
                    expert_mask = (np.array(expert) > 127).astype(np.uint8)

                    st.markdown("### Maska ekspercka")
                    st.image(expert_mask * 255, clamp=True, use_container_width=True)
                    st.markdown("### Obraz z nałożoną maską")
                    st.image(overlay, clamp=True, use_container_width=True)

                    if expert_mask.shape != pred_mask.shape:
                        st.error("Rozmiary maski eksperckiej i wykrytej są różne!")
                    else:
                        st.subheader("Miary porównawcze z maską ekspercką")
                        results = evaluate_mask(expert_mask, pred_mask)

                        # Zbieraj metryki do podsumowania
                        acc_list.append(results['accuracy'])
                        sens_list.append(results['sensitivity'])
                        spec_list.append(results['specificity'])
                        arith_list.append(results['mean_arithmetic'])
                        geom_list.append(results['mean_geometric'])

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
                            index=["AN", "AP"],
                            columns=["PN", "PP"]
                        )

                        conf_matrix_sum += conf_matrix

                        st.markdown("**Macierz pomyłek**")
                        st.dataframe(conf_df.style.format("{:.0f}"))

        st.write("Legenda:")
        st.write("AN - Actual Negative")
        st.write("AP - Actual Positive")
        st.write("PN - Predicted Negative")
        st.write("PP - Predicted Positive")

        # === PODSUMOWANIE WYNIKÓW ===
        if acc_list:
            st.subheader("Podsumowanie zbiorcze wyników (średnie z 5 obrazów):")

            st.markdown(f"""
                - **Średnia accuracy**: {np.mean(acc_list):.4f}  
                - **Średnia sensitivity (Recall)**: {np.mean(sens_list):.4f}  
                - **Średnia specificity**: {np.mean(spec_list):.4f}  
                - **Średnia arytmetyczna (Se + Sp / 2)**: {np.mean(arith_list):.4f}  
                - **Średnia geometryczna (sqrt(Se × Sp))**: {np.mean(geom_list):.4f}
                """)

            # macierz prawdy
            df_summary = pd.DataFrame({
                "Obraz": file,
                "Accuracy": acc_list,
                "Sensitivity": sens_list,
                "Specificity": spec_list,
                "Średnia Arytmetyczna": arith_list,
                "Średnia Geometryczna": geom_list
            })
            st.dataframe(df_summary.style.format({col: "{:.4f}" for col in df_summary.columns if col != "Obraz"}))
        else:
            st.warning("Nie udało się zebrać wyników — brak zgodnych masek.")

        st.markdown("### Sumaryczna macierz pomyłek (łączna dla 5 obrazów):")
        conf_sum_df = pd.DataFrame(
            conf_matrix_sum,
            index=["Actual Negative", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Positive"]
        )
        st.dataframe(conf_sum_df.style.format("{:.0f}"))



    # ======= Trening klasyfikatora i automatyczna walidacja =======
    st.subheader("🔁 Trening i walidacja klasyfikatora")

    if st.button("🔁 Wytrenuj klasyfikator"):
        if len(files) < 2:
            st.error("Za mało obrazów do podziału na trening i test.")
            st.stop()

        generate_classifier(files, IMAGE_DIR, MASK_DIR)
