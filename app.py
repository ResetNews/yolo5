import os
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
from docx import Document
from fpdf import FPDF
import matplotlib.pyplot as plt

# Ordner für die Textdatenbank
DATABASE_FOLDER = "text_database"
os.makedirs(DATABASE_FOLDER, exist_ok=True)

# Funktion: Text aus Datei extrahieren
def extract_text_from_file(file_path):
    try:
        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            return "".join([page.extract_text() for page in reader.pages])
        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            return None
    except Exception as e:
        print(f"Fehler beim Extrahieren des Texts aus Datei: {e}")
        return ""

# Funktion: Sprache erkennen
def detect_language_safe(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Fehler bei der Spracherkennung: {e}")
        return "Unbekannt"

# Funktion: Ähnlichkeit mit Datenbank vergleichen
def compare_with_database_safe(input_text):
    try:
        files = os.listdir(DATABASE_FOLDER)
        if not files:
            print("Keine Dateien in der Datenbank vorhanden.")
            return {}
        texts = [input_text]
        for file in files:
            with open(os.path.join(DATABASE_FOLDER, file), 'r', encoding='utf-8') as f:
                texts.append(f.read())
        vectorizer = TfidfVectorizer().fit_transform(texts)
        vectors = vectorizer.toarray()
        similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        return dict(zip(files, similarities))
    except Exception as e:
        print(f"Fehler beim Vergleich: {e}")
        return {}

# Hauptprogramm
def main():
    print("DEBUG: Starte Plagiatserkennung...")
    try:
        while True:
            print("\nOptionen:")
            print("1 - Text eingeben")
            print("2 - Webseite scrapen")
            print("3 - Datei hochladen")
            print("4 - Beenden")
            option = input("DEBUG: Wähle eine Option: ")

            if option == "1":
                input_text = input("DEBUG: Gib den Text ein: ")
                print(f"DEBUG: Text erhalten ({len(input_text)} Zeichen).")
            elif option == "2":
                url = input("DEBUG: Gib die URL ein: ")
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, "html.parser")
                    paragraphs = soup.find_all("p")
                    input_text = " ".join([p.get_text() for p in paragraphs])
                    print(f"DEBUG: Text von URL erhalten ({len(input_text)} Zeichen).")
                except Exception as e:
                    print(f"DEBUG: Fehler beim Abrufen von URL: {e}")
                    continue
            elif option == "3":
                file_path = input("DEBUG: Datei hochladen (Pfad): ")
                input_text = extract_text_from_file(file_path)
                if not input_text:
                    print("DEBUG: Keine gültigen Inhalte in der Datei gefunden.")
                    continue
            elif option == "4":
                print("DEBUG: Beenden.")
                break
            else:
                print("DEBUG: Ungültige Option.")
                continue

            language = detect_language_safe(input_text)
            print(f"DEBUG: Sprache erkannt: {language}")

            results = compare_with_database_safe(input_text)
            if results:
                for file, similarity in results.items():
                    print(f"DEBUG: Datei: {file}, Ähnlichkeit: {similarity * 100:.2f}%")
            else:
                print("DEBUG: Keine Ähnlichkeiten gefunden.")
    except KeyboardInterrupt:
        print("\nDEBUG: Programm beendet.")
    except Exception as e:
        print(f"DEBUG: Fehler: {e}")

if __name__ == "__main__":
    main()
