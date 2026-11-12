import pandas as pd
import os

# 1. Проверяем, что очищенный файл читается
try:
    df = pd.read_parquet("data/processed/cleanfull.parquet")
    print(f"Успех! Файл cleanfull.parquet прочитан. Строк: {len(df)}")
except FileNotFoundError:
    print("Ошибка: Файл cleanfull.parquet не найден!")

# 2. Убеждаемся, что исходники не изменены (просто смотрим дату изменения)
raw_file = "data/raw/source.csv"
if os.path.exists(raw_file):
    print(f"Исходный файл на месте. Последнее изменение: {os.path.getmtime(raw_file)}")
