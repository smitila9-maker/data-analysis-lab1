import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import time
import os
from dotenv import load_dotenv

load_dotenv()

parquet_path = "data/processed/cleanfull.parquet"

# Строка подключения
url = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
engine = create_engine(url)

print("Чтение Parquet...")
df = pd.read_parquet(parquet_path)

# Справочники (как в SQLite)
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

print("Загрузка справочников...")
airports.to_sql("airports", engine, if_exists="append", index=False)
types.to_sql("aircraft_types", engine, if_exists="append", index=False)

print("Загрузка рейсов (пакетная вставка)...")
start = time.perf_counter()
df_flights.to_sql(
    "flights", engine, if_exists="append", index=False, chunksize=50000, method="multi"
)
end = time.perf_counter()
print(f"Время загрузки: {end - start:.2f} сек.")
