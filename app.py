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

# Funktion 1: Scraping von Webseiten
def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        return text
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen der Webseite: {e}")
        return ""

# Funktion 2: Text aus Project Gutenberg herunterladen
def download_gutenberg_book(book_id, filename):
    file_path = os.path.join(DATABASE_FOLDER, filename)
    if os.path.exists(file_path):
        print(f"Buch {book_id} existiert bereits als {filename}")
        return
    try:
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response.text)
        print(f"Buch {book_id} heruntergeladen und gespeichert als {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Herunterladen des Buches: {e}")

# Funktion 3: Text mit lokaler Datenbank vergleichen
def compare_with_database(input_text):
    try:
        files = os.listdir(DATABASE_FOLDER)
        if not files:
            print("Keine Dateien in der Datenbank vorhanden.")
            return {}
        texts = [input_text]
        for file in files:
            with open(os.path.join(DATABASE_FOLDER, file), 'r', encoding='utf-8') as f:
                texts.append(f.read())
        
        # Berechnung der Ähnlichkeit
        vectorizer = TfidfVectorizer().fit_transform(texts)
        vectors = vectorizer.toarray()
        similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        return dict(zip(files, similarities))
    except Exception as e:
        print(f"Fehler beim Vergleich mit der Datenbank: {e}")
        return {}

# Erweiterung: Sprache erkennen
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Fehler bei der Spracherkennung: {e}")
        return "Unbekannt"

# Erweiterung: PDF-Text extrahieren
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Fehler beim Extrahieren von PDF-Text: {e}")
        return ""

# Erweiterung: Word-Text extrahieren
def extract_text_from_word(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Fehler beim Extrahieren von Word-Text: {e}")
        return ""

# Erweiterung: Ergebnisse visualisieren
def visualize_results(results):
    try:
        files = list(results.keys())
        similarities = [sim * 100 for sim in results.values()]
        plt.barh(files, similarities)
        plt.xlabel("Ähnlichkeit (%)")
        plt.ylabel("Dateien")
        plt.title("Ähnlichkeit mit der Datenbank")
        plt.show()
    except Exception as e:
        print(f"Fehler bei der Visualisierung: {e}")

# Hauptprogramm
def main():
    print("Willkommen zur Plagiatserkennungs-App!")
    try:
        while True:
            # Menüoptionen anzeigen
            print("\nOptionen:")
            print("1 - Text eingeben")
            print("2 - Webseite scrapen")
            print("3 - Datei hochladen")
            print("4 - Beenden")
            option = input("Wähle eine Option: ")

            if option == "1":
                input_text = input("Gib den Text ein: ")
            elif option == "2":
                url = input("Gib die URL der Webseite ein: ")
                input_text = scrape_text_from_url(url)
                if not input_text:
                    print("Kein Text von der Webseite erhalten.")
                    continue
            elif option == "3":
                file_path = input("Gib den Pfad zur Datei ein: ")
                if file_path.endswith(".pdf"):
                    input_text = extract_text_from_pdf(file_path)
                elif file_path.endswith(".docx"):
                    input_text = extract_text_from_word(file_path)
                else:
                    print("Dateiformat nicht unterstützt. Unterstützte Formate: .pdf, .docx")
                    continue
            elif option == "4":
                print("Programm wird beendet.")
                break
            else:
                print("Ungültige Option. Bitte versuche es erneut.")
                continue

            # Sprache erkennen
            language = detect_language(input_text)
            print(f"Erkannte Sprache: {language}")

            # Vergleich mit der Datenbank
            print("Vergleiche mit der lokalen Datenbank...")
            results = compare_with_database(input_text)

            # Ergebnisse anzeigen
            if results:
                for file, similarity in results.items():
                    print(f"Ähnlichkeit mit {file}: {similarity * 100:.2f}%")
                visualize_results(results)
            else:
                print("Keine Texte in der Datenbank gefunden.")
    except KeyboardInterrupt:
        print("\nProgramm wurde manuell beendet.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    main()
