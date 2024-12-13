import joblib
import string
from nltk.stem import WordNetLemmatizer

model_file = 'models/news_classifier_model_30k.pkl'
vectorizer_file = 'models/tfidf_vectorizer_30k.pkl'

lemmatizer = WordNetLemmatizer()


def load_model_and_vectorizer(model_file, vectorizer_file):
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    print(f"Модель и векторизатор загружены из файлов {model_file} и {vectorizer_file}")
    return model, vectorizer


def preprocess_text(text):
    text = text.lower()

    # Удаление знаков препинания
    text = ''.join([char for char in text if char not in string.punctuation])

    return text


model, vectorizer = load_model_and_vectorizer(model_file, vectorizer_file)

new_texts = [
    "Ученые нашли необыкновенные минералы, спрятанные подо льдом арктики",
    "В Москве на следующей неделе ожидаются серьезные осадки в виде дождя"
]

new_texts_cleaned = [preprocess_text(text) for text in new_texts]
new_X = vectorizer.transform(new_texts_cleaned)

new_predictions = model.predict(new_X)
print("Скорее всего, данный текст соответствует категориям:", new_predictions)
