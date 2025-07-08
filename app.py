import streamlit as st
from screen3 import run_screen3
from screen4 import run_screen4
from screen5 import run_screen5
from compare_4_and_5 import run_compare

st.set_page_config(page_title="Wykrywanie naczyń dna oka", layout="wide")
st.title("🩺 Detekcja naczyń krwionośnych")

# Wybór trybu
mode = st.sidebar.selectbox("Wybierz tryb:", [
    "🔹 Wymagania na 3",
    "🔸 Wymagania na 4 (klasyfikator)",
    "⭐ Wymagania na 5 (sieć neuronowa)",
    "⚖️ Porównanie klasyfikatorów"
])

# Przekierowanie do odpowiedniego ekranu
if mode == "🔹 Wymagania na 3":
    run_screen3()
elif mode == "🔸 Wymagania na 4 (klasyfikator)":
    run_screen4()
elif mode == "⭐ Wymagania na 5 (sieć neuronowa)":
    run_screen5()
elif mode == "⚖️ Porównanie klasyfikatorów":
    run_compare()
