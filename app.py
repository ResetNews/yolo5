import os
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
from docx import Document
import schedule
import time
from fpdf import FPDF
import matplotlib.pyplot as plt

# Ordner für die Textdatenbank
DATABASE_FOLDER = "text_database"
os.makedirs(DATABASE_FOLDER, exist_ok=True)

# Funktion 1: Scraping von Webseiten
def scrape_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = " ".join([p.get_text() for p in paragraphs])
    return text

# Funktion 2: Text aus Project Gutenberg herunterladen
def download_gutenberg_book(book_id, filename):
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    response = requests.get(url)
    with open(os.path.join(DATABASE_FOLDER, filename), 'w', encoding='utf-8') as file:
        file.write(response.text)
    print(f"Buch {book_id} heruntergeladen und gespeichert als {filename}")

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
    return detect(text)

# Erweiterung: Semantische Ähnlichkeit
def calculate_semantic_similarity(text1, text2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([text1, text2])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item() * 100

# Erweiterung: PDF-Text extrahieren
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Erweiterung: Word-Text extrahieren
def extract_text_from_word(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

# Erweiterung: Datenbank automatisch aktualisieren
def update_database():
    print("Aktualisiere die Textdatenbank...")
    # Beispiel: Hinzufügen eines weiteren Buches aus Project Gutenberg
    download_gutenberg_book(1661, "sherlock_holmes.txt")  # Sherlock Holmes

schedule.every().day.at("03:00").do(update_database)

# Erweiterung: Plagiatsbericht erstellen
def generate_report(results, output_file="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Plagiatsbericht", ln=True, align="C")
    for file, similarity in results.items():
        pdf.cell(0, 10, txt=f"{file}: {similarity * 100:.2f}%", ln=True)
    pdf.output(output_file)
    print(f"Bericht gespeichert als {output_file}")

# Erweiterung: Ergebnisse visualisieren
def visualize_results(results):
    files = list(results.keys())
    similarities = [sim * 100 for sim in results.values()]
    plt.barh(files, similarities)
    plt.xlabel("Ähnlichkeit (%)")
    plt.ylabel("Dateien")
    plt.title("Ähnlichkeit mit der Datenbank")
    plt.show()

# Hauptprogramm
def main():
    print("Willkommen zur Plagiatserkennungs-App!")
    
    # Text eingeben oder URL scrapen
    option = input("Möchtest du (1) einen Text eingeben, (2) eine Webseite scrapen oder (3) eine Datei hochladen? (1/2/3): ")
    if option == "1":
        input_text = input("Gib den Text ein: ")
    elif option == "2":
        url = input("Gib die URL der Webseite ein: ")
        input_text = scrape_text_from_url(url)
        print("Webseitentext erfolgreich extrahiert.")
    elif option == "3":
        file_path = input("Gib den Pfad zur Datei ein: ")
        if file_path.endswith(".pdf"):
            input_text = extract_text_from_pdf(file_path)
        elif file_path.endswith(".docx"):
            input_text = extract_text_from_word(file_path)
        else:
            print("Dateiformat nicht unterstützt. Unterstützte Formate: .pdf, .docx")
            return
    else:
        print("Ungültige Option. Beende das Programm.")
        return

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

if __name__ == "__main__":
    # Füge Beispielbücher zur Datenbank hinzu (optional)
    if not os.listdir(DATABASE_FOLDER):
        add_example_books()

    # Starte Datenbankaktualisierung im Hintergrund
    schedule.run_all()

    # Starte die App
    main()
