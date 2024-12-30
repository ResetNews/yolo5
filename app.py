import os
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Hauptprogramm
def main():
    print("Willkommen zur Plagiatserkennungs-App!")
    
    # Text eingeben oder URL scrapen
    option = input("Möchtest du (1) einen Text eingeben oder (2) eine Webseite scrapen? (1/2): ")
    if option == "1":
        input_text = input("Gib den Text ein: ")
    elif option == "2":
        url = input("Gib die URL der Webseite ein: ")
        input_text = scrape_text_from_url(url)
        print("Webseitentext erfolgreich extrahiert.")
    else:
        print("Ungültige Option. Beende das Programm.")
        return

    # Vergleich mit Datenbank
    print("Vergleiche mit der lokalen Datenbank...")
    results = compare_with_database(input_text)

    # Ergebnisse anzeigen
    if results:
        for file, similarity in results.items():
            print(f"Ähnlichkeit mit {file}: {similarity * 100:.2f}%")
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

    # Starte die App
    main()
