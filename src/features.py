import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Регулярные выражения для сущностей (адаптировано под позывные)
# Ищем буквенные префиксы (авиакомпании) и цифровые суффиксы (номера рейсов)
PATTERN_PREFIX = re.compile(r"^[A-Z]+")
PATTERN_NUMBERS = re.compile(r"\d+")


# 2. Токенизация (разбиваем позывной на части)
def tokens(text):
    if pd.isna(text):
        return []
    text = str(text).upper()
    # Разбиваем по границе буквы-цифры
    words = re.findall(r"[A-Z]+|\d+", text)
    return [w for w in words if len(w) > 1]  # Убираем одиночные символы


# 3. Частотный словарь и биграммы
def build_vocab(series):
    freq = Counter()
    pairs = Counter()
    for text in series.dropna():
        words = tokens(text)
        freq.update(words)
        pairs.update(zip(words, words[1:]))
    return freq, pairs


# 4. Динамика доли слов по годам/месяцам
def word_share_over_time(df, words, date_col="day"):
    # Приводим даты к году-месяцу
    df_temp = df.dropna(subset=[date_col]).copy()
    df_temp["period"] = df_temp[date_col].dt.to_period("M")

    # Для каждого слова считаем долю записей, где оно есть
    shares = {}
    for word in words:
        # Маска: есть ли слово в позывном
        has_word = (
            df_temp["callsign"].astype(str).str.contains(word, case=False, na=False)
        )
        shares[word] = has_word.groupby(df_temp["period"]).mean()

    return pd.DataFrame(shares)


# 5. TF-IDF для поиска характерных слов для групп
def get_tfidf_top_words(df, text_col, group_col, top_n=5):
    # Убираем пропуски
    df_clean = df.dropna(subset=[text_col, group_col]).copy()

    # Создаем TF-IDF матрицу
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df_clean[text_col].astype(str))

    # Получаем названия признаков (слов)
    feature_names = vectorizer.get_feature_names_out()

    # Считаем средний TF-IDF по группам
    results = {}
    for group in df_clean[group_col].unique()[:3]:  # Берем первые 3 группы
        # --- ИСПРАВЛЕНИЕ: Превращаем Pandas Series в numpy array ---
        mask = (df_clean[group_col] == group).values

        group_mean = tfidf_matrix[mask].mean(axis=0).A1
        top_indices = group_mean.argsort()[-top_n:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        results[group] = top_words

    return results
