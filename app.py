
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Funktion zur Berechnung der Ähnlichkeit
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)
    return similarity[0][1] * 100  # Prozentuale Übereinstimmung

# Streamlit App
st.title("Plagiatserkennungs-App")

st.write("Lade einen Text hoch oder füge ihn unten ein, um Plagiate zu erkennen.")

# Text-Eingabe oder Datei-Upload
uploaded_file = st.file_uploader("Textdatei hochladen", type=["txt"])
input_text = st.text_area("Oder füge deinen Text hier ein:")

# Vergleichstext
comparison_text = st.text_area("Füge den Vergleichstext hier ein:")

if st.button("Analyse starten"):
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
    else:
        text = input_text

    if text and comparison_text:
        similarity = calculate_similarity(text, comparison_text)
        st.write(f"Die Texte sind zu {similarity:.2f}% ähnlich.")
        if similarity > 70:
            st.warning("Plagiatsverdacht!")
        else:
            st.success("Kein Plagiat festgestellt.")
    else:
        st.error("Bitte beide Texte bereitstellen.")
