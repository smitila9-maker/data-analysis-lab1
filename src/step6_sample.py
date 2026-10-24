import pandas as pd
import glob

# Находим все csv файлы в папке data/raw
files = glob.glob("data/raw/*.csv")
print(f"Найдено файлов: {len(files)}")

# Читаем и склеиваем все в один датафрейм
df_list = [pd.read_csv(f) for f in files]
full_df = pd.concat(df_list, ignore_index=True)

# Сохраняем объединенный файл обратно в raw (как исходный монолит)
full_df.to_csv("data/raw/source.csv", index=False)
print(f"Объединено строк: {len(full_df)}")
