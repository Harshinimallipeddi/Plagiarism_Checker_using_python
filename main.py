import nltk
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=UserWarning)

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text.lower())

    filtered_words = [wordnet_lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(filtered_words)

def calculate_similarity(text1, text2):
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text1, preprocessed_text2])

    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity_score * 100, 2)

def check_plagiarism(file1, file2, threshold=80):
    with open(file1, 'r', encoding='utf-8') as f1:
        text1 = f1.read()

    with open(file2, 'r', encoding='utf-8') as f2:
        text2 = f2.read()

    similarity_score = calculate_similarity(text1, text2)
    if similarity_score >= threshold:
        return "Plagiarized", similarity_score
    else:
        return "Not Plagiarized", similarity_score

file1 = "file1.txt"
file2 = "file2.txt"

result, similarity_score = check_plagiarism(file1, file2)
print(result)
print("Similarity Score:", "{:.2f}".format(similarity_score), "%")
