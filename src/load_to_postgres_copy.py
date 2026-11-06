import pandas as pd
import psycopg2
import time
import os
from dotenv import load_dotenv
from io import StringIO

load_dotenv()

parquet_path = "data/processed/cleanfull.parquet"

# Сначала подготовим CSV-файлы для COPY (или используем StringIO)
print("Чтение Parquet...")
df = pd.read_parquet(parquet_path)

# Подготовка справочников
airports = pd.DataFrame(
    {"airport_code": pd.concat([df["origin"], df["destination"]]).dropna().unique()}
)
airports["airport_name"] = airports["airport_code"]
types = (
    df[["typecode"]]
    .dropna()
    .drop_duplicates()
    .rename(columns={"typecode": "type_code"})
)
types["description"] = types["type_code"]

cols = [
    "callsign",
    "aircraft_uid",
    "typecode",
    "origin",
    "destination",
    "firstseen",
    "lastseen",
    "day",
    "latitude_1",
    "longitude_1",
    "altitude_1",
    "latitude_2",
    "longitude_2",
    "altitude_2",
]
df_flights = df[cols]

# Подключение
conn = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    dbname=os.getenv("PG_DB"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
)
cur = conn.cursor()


def copy_from_df(df, table_name):
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False, na_rep="\\N")
    buffer.seek(0)
    # Явно перечисляем столбцы, чтобы пропустить id
    columns = (
        "callsign",
        "aircraft_uid",
        "typecode",
        "origin",
        "destination",
        "firstseen",
        "lastseen",
        "day",
        "latitude_1",
        "longitude_1",
        "altitude_1",
        "latitude_2",
        "longitude_2",
        "altitude_2",
    )
    cur.copy_from(buffer, table_name, sep=",", columns=columns)
    conn.commit()


print("Загрузка справочников...")
# Для справочников можно использовать обычный INSERT (они маленькие)
for _, row in airports.iterrows():
    cur.execute(
        "INSERT INTO airports (airport_code, airport_name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (row["airport_code"], row["airport_name"]),
    )
for _, row in types.iterrows():
    cur.execute(
        "INSERT INTO aircraft_types (type_code, description) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (row["type_code"], row["description"]),
    )
conn.commit()

print("Загрузка рейсов (COPY)...")
start = time.perf_counter()
copy_from_df(df_flights, "flights")
end = time.perf_counter()
print(f"Время загрузки через COPY: {end - start:.2f} сек.")

cur.close()
conn.close()
