# 1 блок

import pandas as pd
import numpy as np
import logging

# Настраиваем логгер, чтобы он писал в консоль (и в файл, как требует методичка)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 2 блок


def normalize_columns(df):
    """Приводит имена столбцов к нижнему регистру, заменяет пробелы на подчеркивания."""
    logger.info("Нормализация имен столбцов...")
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def cast_types(df):
    """Явное приведение типов и конвертация дат."""
    logger.info("Приведение типов данных...")
    # Схему мы делали в Шаге 10
    schema = {
        "callsign": "category",
        "typecode": "category",
        "origin": "category",
        "destination": "category",
        "latitude_1": "float32",
        "longitude_1": "float32",
        "altitude_1": "float32",
        "latitude_2": "float32",
        "longitude_2": "float32",
        "altitude_2": "float32",
        "aircraft_uid": "string",
    }
    # Применяем типы, если столбцы существуют
    cols_to_cast = {k: v for k, v in schema.items() if k in df.columns}
    df = df.astype(cols_to_cast)

    # Даты (errors='coerce' превратит нераспознанные строки в NaT)
    date_cols = ["firstseen", "lastseen", "day"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# 3 блок


def handle_hidden_nulls(df):
    """Заменяет скрытые пропуски (строки 'нет данных', нули и т.д.) на NaN."""
    logger.info("Поиск и замена скрытых пропусков...")
    # Список того, что считаем пропуском
    na_values = ["нет данных", "nan", "none", "null", "", "n/a", "-"]

    # Для текстовых столбцов заменяем строки на NaN
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        # Если это категория, сначала приводим к строке, чтобы заменить
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str).replace(na_values, np.nan).astype("category")
        else:
            df[col] = df[col].replace(na_values, np.nan)

    return df


def analyze_missing(df):
    """Выводит долю пропусков (для отчета) и удаляет строки, где пропущены критически важные данные."""
    logger.info("Анализ структуры пропусков...")
    missing_pct = df.isna().sum() / len(df) * 100
    print("\nДоля пропусков по столбцам (%):")
    print(missing_pct[missing_pct > 0].sort_values(ascending=False))

    # Стратегия: если пропущен аэропорт вылета/прилета, такой полет бесполезен
    # Удаляем строки, где origin или destination пустые
    initial_len = len(df)
    df = df.dropna(subset=["origin", "destination"])
    logger.info(f"Удалено строк с пустыми origin/destination: {initial_len - len(df)}")

    return df


# 4 блок


def drop_duplicates(df):
    """Удаление полных дубликатов и дубликатов по ключу."""
    logger.info("Удаление дубликатов...")
    initial_len = len(df)

    # 1. Полные дубликаты
    df = df.drop_duplicates()
    logger.info(f"Удалено полных дубликатов: {initial_len - len(df)}")

    # 2. Дубликаты по ключу (например, один и тот же самолет в одно и то же время)
    # Оставляем последнюю запись
    initial_len = len(df)
    df = df.drop_duplicates(subset=["aircraft_uid", "firstseen"], keep="last")
    logger.info(f"Удалено дубликатов по ключу: {initial_len - len(df)}")

    return df


def normalize_categories(df):
    """Нормализация текстовых категорий: регистр, пробелы, объединение редких."""
    logger.info("Нормализация категориальных столбцов...")
    cat_cols = ["origin", "destination", "typecode"]

    for col in cat_cols:
        if col in df.columns:
            # Приводим к строке, убираем пробелы, делаем верхний регистр
            # (Аэропорты обычно пишутся заглавными: SVO, JFK)
            df[col] = df[col].astype(str).str.strip().str.upper()

            # Объединение редких категорий (если категория встречается реже 10 раз)
            counts = df[col].value_counts()
            rare_cats = counts[counts < 10].index
            if len(rare_cats) > 0:
                df[col] = df[col].replace(rare_cats, "OTHER")

            # Возвращаем тип category
            df[col] = df[col].astype("category")

    return df


# 5 блок


def clean_text_columns(df):
    """Очистка текста: удаление разметки, лишних пробелов."""
    logger.info("Очистка текстовых столбцов...")
    # Предположим, что callsign - это текст. Убираем лишние пробелы.
    if "callsign" in df.columns:
        df["callsign"] = (
            df["callsign"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        )
        # Заменяем 'nan' обратно на NaN
        df["callsign"] = df["callsign"].replace("nan", np.nan).astype("category")
    return df


def fix_outliers(df):
    """Выявление и обработка выбросов (высоты)."""
    logger.info("Обработка выбросов...")

    # Метод 1: Предметное ограничение (высота не может быть отрицательной)
    # или выше 15 000 метров (примерно 50 000 футов)
    mask_invalid = (df["altitude_1"] < 0) | (df["altitude_1"] > 50000)
    logger.info(
        f"Найдено некорректных высот (правило предметной области): {mask_invalid.sum()}"
    )

    # Заменяем их на NaN (потом можно заполнить медианой)
    df.loc[mask_invalid, "altitude_1"] = np.nan

    # Метод 2: Статистический (IQR) для сравнения (по методичке нужно 2 способа)
    Q1 = df["altitude_1"].quantile(0.25)
    Q3 = df["altitude_1"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = (
        (df["altitude_1"] < lower_bound) | (df["altitude_1"] > upper_bound)
    ).sum()
    logger.info(
        f"Найдено выбросов по правилу IQR: {outliers_iqr} (оставляем их, так как они могут быть реальными)"
    )

    return df


# 6 блок (Дополнение)


def check_business_rules(df):
    """Проверка предметных ограничений (даты в будущем, некорректные диапазоны)."""
    logger.info("Проверка бизнес-правил...")

    # 1. Проверка дат в будущем (если дата больше сегодняшней)
    today = pd.Timestamp.now(tz="UTC")
    if "firstseen" in df.columns:
        future_dates = (df["firstseen"] > today).sum()
        logger.info(f"Найдено дат в будущем (firstseen): {future_dates}")
        # Удаляем строки с датами из будущего
        df = df[df["firstseen"] <= today]

    # 2. Проверка диапазонов координат (широта от -90 до 90, долгота от -180 до 180)
    if "latitude_1" in df.columns:
        invalid_lat = ((df["latitude_1"] < -90) | (df["latitude_1"] > 90)).sum()
        logger.info(f"Найдено некорректных широт (latitude_1): {invalid_lat}")
        # Очищаем их (ставим NaN)
        df.loc[(df["latitude_1"] < -90) | (df["latitude_1"] > 90), "latitude_1"] = (
            np.nan
        )

    return df
