import pandas as pd
import sqlite3
import time
import os

# Пути к файлам
parquet_path = "data/processed/cleanfull.parquet"
db_path = "data/processed/dataset.db"


def load_data():
    print("Чтение очищенного Parquet файла...")
    df = pd.read_parquet(parquet_path)

    # 1. Готовим справочники
    print("Подготовка справочников...")
    airports = pd.DataFrame(
        {"airport_code": pd.concat([df["origin"], df["destination"]]).dropna().unique()}
    )
    airports["airport_name"] = airports["airport_code"]

    # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
    types = df[["typecode"]].dropna().drop_duplicates()
    types = types.rename(
        columns={"typecode": "type_code"}
    )  # Переименовываем в type_code
    types["description"] = types["type_code"]
    # -------------------------

    # Оставляем только нужные столбцы для основной таблицы
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

    print("Подключение к SQLite...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Очищаем таблицы (на случай, если там что-то есть)
    cursor.execute("DELETE FROM flights")
    cursor.execute("DELETE FROM airports")
    cursor.execute("DELETE FROM aircraft_types")

    try:
        print("Загрузка справочников (airports, aircraft_types)...")
        airports.to_sql("airports", conn, if_exists="append", index=False)
        types.to_sql("aircraft_types", conn, if_exists="append", index=False)

        print("Загрузка рейсов (flights) пачками по 50 000...")
        start_time = time.perf_counter()

        # Пакетная загрузка (chunksize)
        df_flights.to_sql(
            "flights", conn, if_exists="append", index=False, chunksize=50000
        )

        end_time = time.perf_counter()
        print(f"Загрузка завершена! Время: {end_time - start_time:.2f} сек.")

        # Проверка
        cursor.execute("SELECT COUNT(*) FROM flights")
        print(f"Всего строк в базе: {cursor.fetchone()[0]}")

    except Exception as e:
        import traceback

        traceback.print_exc()
    finally:
        conn.commit()
        conn.close()


if __name__ == "__main__":
    load_data()
# Загрузка завершена! Время: 60.31 сек.
# Всего строк в базе: 7248316
