import pandas as pd
import os
import sys
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
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    input_path = "data/interim/full_data_optimized_full.parquet"  # Откуда берем
    output_path = "data/processed/cleanfull.parquet"  # Куда сохраняем

    # Создаем папку processed, если ее нет
    os.makedirs("data/processed", exist_ok=True)

    logger.info("Загрузка исходных данных...")
    # Для теста читаем первые 500 000 строк.
    # Когда будешь готова прогнать все 14 млн, просто удали .head(500_000)
    df = pd.read_parquet(input_path)

    initial_rows = len(df)
    logger.info(f"Строк до очистки: {initial_rows}")

    # Пошаговый конвейер очистки
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

    logger.info(f"Сохранение очищенного набора в {output_path}...")
    df.to_parquet(output_path)
    logger.info("Готово!")


if __name__ == "__main__":
    main()

# В ходе выполнения скрипта очистки (pipeline.py) на полном наборе данных был обработан 14 009 851 строка. В результате применения правил очистки итоговый набор сократился до 7 248 316 строк.

# Итоговые потери составили 6 761 535 строк (48.26% от исходного объема).

# Такое значительное сокращение требует подробного обоснования. Оно не является ошибкой или потерей важных данных, а обусловлено спецификой датасета и строгими бизнес-правилами:

# Удаление строк с отсутствующими ключевыми признаками (6 760 359 строк): Столбцы origin (аэропорт вылета) и destination (аэропорт прилета) содержали 31% и 24.7% пропусков соответственно. Поскольку целью анализа является изучение перелетов, запись без указания пункта отправления или назначения не несет аналитической ценности. Удаление этих строк — единственно верная стратегия, так как заполнить аэропорт искусственно невозможно.

# Удаление дубликатов (1 176 строк): Были удалены полные дубликаты (66 шт.) и дубликаты по ключу «бортовой номер + время» (1110 шт.), чтобы избежать двойного учета рейсов.

# Остальные потери были связаны с приведением дат (даты в будущем отсутствовали) и некорректными координатами (их также не оказалось).
