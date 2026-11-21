import pandas as pd
import numpy as np
import os
import argparse
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


def build_feature_store(df):
    """Собирает все признаки в одну таблицу."""
    print("Сборка витрины признаков...")
    features = pd.DataFrame(index=df.index)

    # 1. Категориальные признаки (частотное кодирование)
    # Доля записей для каждого аэропорта вылета и типа самолета
    features["origin_freq"] = df["origin"].map(
        df["origin"].value_counts(normalize=True)
    )
    features["typecode_freq"] = df["typecode"].map(
        df["typecode"].value_counts(normalize=True)
    )

    # 2. Временные признаки (из столбца day)
    if "day" in df.columns:
        features["year"] = df["day"].dt.year
        features["month"] = df["day"].dt.month
        features["day_of_week"] = df["day"].dt.dayofweek
        features["is_weekend"] = (df["day"].dt.dayofweek >= 5).astype(int)
        features["days_since_start"] = (df["day"] - df["day"].min()).dt.days

    # 3. Агрегаты по группам (по аэропорту вылета)
    # Число записей в группе
    features["origin_count"] = df["origin"].map(df["origin"].value_counts())
    # Медиана высоты внутри группы
    median_alt = df.groupby("origin")["altitude_1"].transform("median")
    features["origin_median_alt"] = median_alt
    # Отклонение высоты записи от медианы группы
    features["altitude_deviation"] = df["altitude_1"] - median_alt

    # 4. Текстовые признаки (из callsign)
    callsign_str = df["callsign"].astype(str)
    features["callsign_len"] = callsign_str.str.len()
    features["callsign_has_digits"] = callsign_str.str.contains(r"\d").astype(int)
    # Извлекаем префикс (авиакомпанию) и кодируем его частотой
    prefix = callsign_str.str.extract(r"^([A-Za-z]+)", expand=False).str.upper()
    features["airline_freq"] = prefix.map(prefix.value_counts(normalize=True))
    # --- НОВОЕ: Длина префикса авиакомпании ---
    features["airline_prefix_len"] = prefix.str.len().fillna(0).astype(int)

    # 5. Исходные числовые (просто копируем, они пригодятся для моделей)
    features["latitude_1"] = df["latitude_1"]
    features["longitude_1"] = df["longitude_1"]
    features["altitude_1"] = df["altitude_1"]
    # --- НОВОЕ: Координаты и высота второй точки ---
    features["latitude_2"] = df["latitude_2"]
    features["longitude_2"] = df["longitude_2"]
    features["altitude_2"] = df["altitude_2"]

    # --- Отбор признаков ---
    # Убираем почти постоянные столбцы (где >95% одного значения)
    # В нашем случае это может быть is_weekend или year (если данные за один год)
    for col in features.columns:
        if features[col].nunique() <= 1 or (
            features[col].value_counts(normalize=True).iloc[0] > 0.95
        ):
            features.drop(columns=[col], inplace=True)
            print(f"Удален почти постоянный признак: {col}")

    # Добавляем целевую переменную (для обучения)
    features["target"] = df["altitude_1"]

    print(f"Итоговое число колонок: {len(features.columns)}")
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/cleanfull.parquet")
    parser.add_argument("--output", type=str, default="data/processed/features.parquet")
    args = parser.parse_args()

    print(f"Чтение {args.input}...")
    df = pd.read_parquet(args.input)

    features = build_feature_store(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    features.to_parquet(args.output)
    print(f"Витрина сохранена в {args.output}")


if __name__ == "__main__":
    main()
