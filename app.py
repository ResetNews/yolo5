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
    files = os.listdir(DATABASE_FOLDER)
    texts = [input_text]
    for file in files:
        with open(os.path.join(DATABASE_FOLDER, file), 'r', encoding='utf-8') as f:
            texts.append(f.read())
    
    # Berechnung der Ähnlichkeit
    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return dict(zip(files, similarities))

# Erweiterung: Sprache erkennen
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Fehler bei der Spracherkennung: {e}")
        return "Unbekannt"

# Erweiterung: Semantische Ähnlichkeit
def calculate_semantic_similarity(text1, text2):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([text1, text2])
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return similarity.item() * 100
    except Exception as e:
        print(f"Fehler bei der semantischen Ähnlichkeitsberechnung: {e}")
        return 0

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

# Erweiterung: Plagiatsbericht erstellen
def generate_report(results, output_file="report.pdf"):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Plagiatsbericht", ln=True, align="C")
        for file, similarity in results.items():
            pdf.cell(0, 10, txt=f"{file}: {similarity * 100:.2f}%", ln=True)
        pdf.output(output_file)
        print(f"Bericht gespeichert als {output_file}")
    except Exception as e:
        print(f"Fehler beim Erstellen des Berichts: {e}")

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
    while True:
        # Text eingeben oder URL scrapen
        option = input("Möchtest du (1) einen Text eingeben, (2) eine Webseite scrapen, (3) eine Datei hochladen oder (4) beenden? (1/2/3/4): ")
        if option == "1":
            input_text = input("Gib den Text ein: ")
        elif option == "2":
            url = input("Gib die URL der Webseite ein: ")
            input_text = scrape_text_from_url(url)
            if not input_text:
                print("Kein Text von der Webseite erhalten.")
                continue
            print("Webseitentext erfolgreich extrahiert.")
        elif option == "3":
            file_path = input("Gib den Pfad zur Datei ein: ")
            if file_path.endswith(".pdf"):
                input_text = extract_text_from_pdf(file_path)
            elif file_path.endswith(".docx"):
                input_text = extract_text_from_word(file_path)
            else:
                print("Dateiformat nicht unterstützt. Unterstützte Formate: .pdf, .docx")
                continue
            if not input_text:
                print("Fehler beim Verarbeiten der Datei.")
                continue
        elif option == "4":
            print("Programm beendet.")
            break
        else:
            print("Ungültige Option. Bitte versuche es erneut.")
            continue

        # Sprache erkennen
        language = detect_language(input_text)
        print(f"Erkannte Sprache: {language}")

        # Vergleich mit Datenbank
        print("Vergleiche mit der lokalen Datenbank...")
        results = compare_with_database(input_text)

        # Ergebnisse anzeigen
        if results:
            for file, similarity in results.items():
                print(f"Ähnlichkeit mit {file}: {similarity * 100:.2f}%")
            generate_report(results)
            visualize_results(results)
        else:
            print("Keine Texte in der Datenbank gefunden.")

        # Möglichkeit, neue Texte zur Datenbank hinzuzufügen
        save_option = input("Möchtest du diesen Text zur Datenbank hinzufügen? (ja/nein): ")
        if save_option.lower() == "ja":
            filename = input("Gib einen Namen für die Datei an (z.B. text1.txt): ")
            with open(os.path.join(DATABASE_FOLDER, filename), 'w', encoding='utf-8') as f:
                f.write(input_text)
            print("Text erfolgreich gespeichert.")

# Beispiel: Text aus Project Gutenberg hinzufügen
def add_example_books():
    print("Beispieltexte aus Project Gutenberg hinzufügen...")
    download_gutenberg_book(84, "frankenstein.txt")
    download_gutenberg_book(1342, "pride_and_prejudice.txt")
    download_gutenberg_book(1661, "sherlock_holmes.txt")

if __name__ == "__main__":
    # Füge Beispielbücher zur Datenbank hinzu (optional)
    add_example_books()

    # Starte die App
    try:
        main()
    except KeyboardInterrupt:
        print("Programm beendet.")
