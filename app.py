import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Database folder for text files
DATABASE_FOLDER = "text_database"
os.makedirs(DATABASE_FOLDER, exist_ok=True)

# Compare input text with database
def compare_with_database(input_text):
    try:
        files = os.listdir(DATABASE_FOLDER)
        if not files:
            print("No files in the database.")
            return {}

        texts = [input_text]
        for file in files:
            with open(os.path.join(DATABASE_FOLDER, file), 'r', encoding='utf-8') as f:
                texts.append(f.read())

        # Calculate similarity
        vectorizer = TfidfVectorizer().fit_transform(texts)
        vectors = vectorizer.toarray()
        similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        return dict(zip(files, similarities))
    except Exception as e:
        print(f"Error comparing with the database: {e}")
        return {}

# Main function
def main():
    print("Welcome to the simplified plagiarism checker!")
    try:
        # Input text
        input_text = input("Enter text to check for plagiarism: ")
        if not input_text.strip():
            print("No text provided. Exiting...")
            return

        # Compare with database
        results = compare_with_database(input_text)
        if results:
            for file, similarity in results.items():
                print(f"Similarity with {file}: {similarity * 100:.2f}%")
        else:
            print("No matches found in the database.")

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
