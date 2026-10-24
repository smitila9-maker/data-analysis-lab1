import pandas as pd
import os
import argparse
import logging
from src.cleaning import (
    normalize_columns,
    cast_types,
    handle_hidden_nulls,
    analyze_missing,
    drop_duplicates,
    normalize_categories,
    clean_text_columns,
    fix_outliers,
    check_business_rules,
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Настраиваем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Пайплайн очистки данных о полетах")
    parser.add_argument(
        "--input", type=str, required=True, help="Путь к исходному parquet файлу"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Путь для сохранения результата"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "sample"],
        default="sample",
        help="Режим: full (все данные) или sample (первые 500к строк)",
    )
    args = parser.parse_args()

    # Проверка, что исходный файл существует
    if not os.path.exists(args.input):
        logger.error(f"Файл не найден: {args.input}")
        return

    logger.info(f"Загрузка данных из {args.input}...")
    df = pd.read_parquet(args.input)

    if args.mode == "sample":
        df = df.head(500_000)
        logger.info("Режим SAMPLE: взяты первые 500 000 строк.")

    initial_rows = len(df)
    logger.info(f"Строк до очистки: {initial_rows}")

    # Конвейер очистки
    df = normalize_columns(df)
    df = cast_types(df)
    df = handle_hidden_nulls(df)
    df = analyze_missing(df)
    df = drop_duplicates(df)
    df = normalize_categories(df)
    df = clean_text_columns(df)
    df = fix_outliers(df)
    df = check_business_rules(df)

    final_rows = len(df)
    logger.info(f"Строк после очистки: {final_rows}")
    logger.info(
        f"Потеряно строк: {initial_rows - final_rows} ({(1 - final_rows / initial_rows) * 100:.2f}%)"
    )

    # Создаем папку для вывода, если ее нет
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info(f"Сохранение очищенного набора в {args.output}...")
    df.to_parquet(args.output)
    logger.info("Готово!")


if __name__ == "__main__":
    main()
