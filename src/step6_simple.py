import pandas as pd

input_file = "data/raw/source.csv"  # Имя твоего исходного файла
output_file = "data/interim/sample_200k.csv"  # Куда сохраним срез
sample_size = 200_000

print("Начинаем чтение и создание среза...")

# Читаем файл порциями (чанками), пока не наберем 200 000 строк
chunks = pd.read_csv(input_file, chunksize=50_000)
sample_df = pd.concat(
    [chunk for chunk in chunks if len(chunk) > 0][:4]
)  # Берем первые 4 чанка по 50к = 200к

# Если в файле меньше 200 000 строк, возьмем столько, сколько есть
sample_df = sample_df.head(sample_size)

# Сохраняем в папку interim
sample_df.to_csv(output_file, index=False)

print(f"Готово! Срез сохранен в {output_file}")
print(f"Количество строк в срезе: {len(sample_df)}")
