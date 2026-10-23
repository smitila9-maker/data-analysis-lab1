import pandas as pd
import os

# Укажи путь к твоему файлу (замени source.csv на свое имя файла, если оно другое)
file_path = "data/raw/source.csv"

# 1. Размер файла в мегабайтах
size_mb = os.path.getsize(file_path) / (1024 * 1024)
print(f"Размер файла: {size_mb:.2f} МБ")

# 2. Читаем только первые 100 строк, чтобы узнать структуру и типы
print("\n--- Первые 100 строк ---")
df_head = pd.read_csv(file_path, nrows=100)

# 3. Выводим перечень столбцов и типы данных
print("\n--- Перечень столбцов и их типы ---")
print(df_head.dtypes)

# 4. Посмотрим на сами данные (первые 5 строк)
print("\n--- Первые 5 строк данных ---")
print(df_head.head())
