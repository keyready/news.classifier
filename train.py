import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, accuracy_score
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json
import joblib  # Для сохранения и загрузки модели

# Функция для загрузки данных из JSON
def load_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Функция для сохранения модели и векторизатора
def save_model_and_vectorizer(model, vectorizer, model_file, vectorizer_file):
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    print(f"Модель и векторизатор сохранены в файлы {model_file} и {vectorizer_file}")

# Функция для загрузки модели и векторизатора
def load_model_and_vectorizer(model_file, vectorizer_file):
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    print(f"Модель и векторизатор загружены из файлов {model_file} и {vectorizer_file}")
    return model, vectorizer

# Путь к файлам
file_path = 'output_data.json'  # Путь к файлу с данными
model_file = 'models/news_classifier_model_30k.pkl'  # Файл для сохранения модели
vectorizer_file = 'models/tfidf_vectorizer_30k.pkl'  # Файл для сохранения векторизатора

# Загружаем данные из JSON
data = load_data_from_json(file_path)

# Создаем DataFrame
df = pd.DataFrame(data)
df = df.sample(n=30000, random_state=42)  # Оставить только 10,000 строк
print(df)

# Загружаем необходимые ресурсы из nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Стоп-слова и лемматизатор
stop_words = set(stopwords.words('russian'))
lemmatizer = WordNetLemmatizer()

# Функция для очистки текста
def preprocess_text(text):
    if text is None or pd.isna(text):
        return ''  # Обрабатываем None и NaN как пустые строки

    # Приведение текста в нижний регистр
    text = text.lower()

    # Удаление знаков препинания
    text = ''.join([char for char in text if char not in string.punctuation])

    # Лемматизация и удаление стоп-слов
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])

    return text

# Удаляем строки, где 'text' пустой или None
df = df[df['text'].notna() & (df['text'] != '')]

# Применяем предобработку
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Векторизация текста с использованием TF-IDF
vectorizer = TfidfVectorizer()

# Преобразуем очищенные тексты в матрицу признаков
X = vectorizer.fit_transform(df['cleaned_text'])
print(X.shape)  # Размер матрицы признаков

# Категории для классификации
y = df['category']

# Разделение данных на обучающую и тестовую выборки (80% - обучающая, 20% - тестовая)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем модель логистической регрессии
model = LogisticRegression(max_iter=1000)

# Обучаем модель
model.fit(X_train, y_train)

# Сохраняем модель и векторизатор, чтобы не переобучать каждый раз
save_model_and_vectorizer(model, vectorizer, model_file, vectorizer_file)

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test)

# Оценка точности модели
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Подробный отчет по меткам
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Прогнозирование для новых текстов
new_texts = [
    "на украине все плохо",
]

# Применяем предобработку и векторизацию
new_texts_cleaned = [preprocess_text(text) for text in new_texts]
new_X = vectorizer.transform(new_texts_cleaned)

# Прогнозируем категории для новых текстов
new_predictions = model.predict(new_X)
print("Predictions for new texts:", new_predictions)
