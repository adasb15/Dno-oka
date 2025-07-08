import streamlit as st
import os
import pandas as pd
from PIL import Image
from utils.processing import *

def run_screen3():
    # === ŚCIEŻKI ===
    IMAGE_DIR = "stare-images"
    MASK_DIR = "labels-ah"

    # === LISTA OBRAZÓW DO WYBORU ===
    available_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".ppm"))
    ])
    selected_file = st.selectbox("Wybierz obraz z bazy", available_files, index=None)

    # === PARAMETRY ===
    apply_postprocessing = st.checkbox("Zastosuj końcowe przetwarzanie maski (oczyszczanie)", value=True)
    show_fov = st.checkbox("Pokaż wygenerowany FOV", value=True)
    show_expert_mask = st.checkbox("Pokaż maske ekspercką", value=True)

    # === WCZYTYWANIE I PRZETWARZANIE ===
    if selected_file:
        image_path = os.path.join(IMAGE_DIR, selected_file)
        mask_path = os.path.join(MASK_DIR, os.path.splitext(selected_file)[0] + ".ah.ppm")

        # Wczytaj obraz
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.array(image)

        st.subheader("Wyniki przetwarzania wybranego obrazu")
        with st.spinner("Przetwarzanie..."):
            preprocessed = preprocess_image(image_np)

            mask_fov = generate_fov_mask_by_brightness(image_np)

            vessel_mask = apply_frangi_filter(preprocessed / 255, mask_fov)

            #vessel_mask_copy = vessel_mask.copy()

            if apply_postprocessing:
                vessel_mask = postprocess_mask(vessel_mask * 255)

        # === WIZUALIZACJA ===
        overlay = visualize_overlay(image_np, vessel_mask)

        if show_fov and show_expert_mask:
            col1, col2, col3, col4, col5 = st.columns(5)
        elif show_fov or show_expert_mask:
            col1, col2, col3, col4 = st.columns(4)
        else:
            col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div style='min-height:100px'><h3>Oryginalny obraz</h3></div>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("<div style='min-height:100px'><h3>Maska naczyń (binarna)</h3></div>", unsafe_allow_html=True)
            st.image(vessel_mask * 255, clamp=True, use_container_width=True)

        if show_expert_mask and not show_fov:
            with col3:
                st.markdown("<div style='min-height:100px'><h3>Maska ekspercka</h3></div>", unsafe_allow_html=True)
                st.image(Image.open(mask_path).convert("L"), clamp=True, use_container_width=True)
            with col4:
                st.markdown("<div style='min-height:100px'><h3>Obraz z nałożoną maską</h3></div>",
                            unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)

        elif not show_expert_mask and show_fov:
            with col3:
                st.markdown("<div style='min-height:100px'><h3>Wygenerowany FOV</h3></div>", unsafe_allow_html=True)
                st.image(mask_fov * 255, clamp=True, use_container_width=True)
            with col4:
                st.markdown("<div style='min-height:100px'><h3>Obraz z nałożoną maską</h3></div>", unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)

        elif show_fov and show_expert_mask:
            with col3:
                st.markdown("<div style='min-height:100px'><h3>Maska ekspercka</h3></div>", unsafe_allow_html=True)
                st.image(Image.open(mask_path).convert("L"), clamp=True, use_container_width=True)
            with col4:
                st.markdown("<div style='min-height:100px'><h3>Wygenerowany FOV</h3></div>", unsafe_allow_html=True)
                st.image(mask_fov * 255, clamp=True, use_container_width=True)
            with col5:
                st.markdown("<div style='min-height:100px'><h3>Obraz z nałożoną maską</h3></div>",
                            unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)

        elif not (show_fov and show_expert_mask):
            with col3:
                st.markdown("<div style='min-height:100px'><h3>Obraz z nałożoną maską</h3></div>",
                            unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)

        # === EWALUACJA ===
        if os.path.exists(mask_path):
            expert = Image.open(mask_path).convert("L")
            expert_mask = (np.array(expert) > 127).astype(np.uint8)

            if expert_mask.shape != vessel_mask.shape:
                st.error("Rozmiary maski eksperckiej i wykrytej są różne!")
            else:
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
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_np = np.array(image)

                with st.spinner("Przetwarzanie..."):
                    preprocessed = preprocess_image(image_np)
                    mask_fov = generate_fov_mask_by_brightness(image_np)

                    vessel_mask = apply_frangi_filter(preprocessed / 255, mask_fov)

                    if apply_postprocessing:
                        vessel_mask = postprocess_mask(vessel_mask * 255)

                    # === WIZUALIZACJA ===
                overlay = visualize_overlay(image_np, vessel_mask)

                st.markdown("### Oryginalny obraz")
                st.image(image, use_container_width=True)

                st.markdown("### Maska naczyń (binarna)")
                st.image(vessel_mask * 255, clamp=True, use_container_width=True)

                # === EWALUACJA ===
                if os.path.exists(mask_path):
                    expert = Image.open(mask_path).convert("L")
                    expert_mask = (np.array(expert) > 127).astype(np.uint8)

                    if show_expert_mask:
                        st.markdown("### Maska ekspercka")
                        st.image(Image.open(mask_path).convert("L"), clamp=True, use_container_width=True)

                    if show_fov:
                        st.markdown("### Wygenerowany FOV")
                        st.image(mask_fov * 255, clamp=True, use_container_width=True)

                    st.markdown("### Obraz z nałożoną maską")
                    st.image(overlay, use_container_width=True)

                    if expert_mask.shape != vessel_mask.shape:
                        st.error("Rozmiary maski eksperckiej i wykrytej są różne!")
                    else:
                        st.subheader("Miary porównawcze z maską ekspercką")
                        results = evaluate_mask(expert_mask, vessel_mask)

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




