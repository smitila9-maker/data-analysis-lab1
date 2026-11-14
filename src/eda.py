import pandas as pd
import numpy as np


def get_anomaly_masks(df, column="altitude_1"):
    """
    Возвращает две маски аномалий и их пересечение.
    Правило 1: Межквартильный размах (IQR).
    Правило 2: Предметное ограничение (высота < 0).
    """
    # --- Правило 1: IQR ---
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Маска для IQR (True там, где выброс)
    mask_iqr = (df[column] < lower_bound) | (df[column] > upper_bound)

    # --- Правило 2: Предметное ограничение (высота меньше 0) ---
    # В авиации высота не может быть отрицательной (относительно уровня моря)
    mask_business = df[column] < 0

    # --- Пересечение (обе маски True) ---
    mask_intersection = mask_iqr & mask_business

    return mask_iqr, mask_business, mask_intersection


def print_anomaly_stats(df, mask_iqr, mask_business, mask_intersection):
    """Выводит статистику для таблицы 4."""
    total = len(df)
    count_iqr = mask_iqr.sum()
    count_business = mask_business.sum()
    count_intersection = mask_intersection.sum()

    print(f"Всего строк: {total}")
    print(f"Межквартильный размах (IQR): {count_iqr} ({count_iqr / total * 100:.2f}%)")
    print(
        f"Второе правило (высота < 0): {count_business} ({count_business / total * 100:.2f}%)"
    )
    print(
        f"Пересечение: {count_intersection} ({count_intersection / total * 100:.2f}%)"
    )
import time

def mean_loop(values):
    """Считает среднее с помощью обычного цикла Python (игнорируя NaN)."""
    total = 0.0
    count = 0
    for value in values:
        if value == value: # Проверка на NaN (пропуск не равен сам себе)
            total += float(value)
            count += 1
    return total / count

def mean_numpy(values):
    """Считает среднее с помощью NumPy (игнорируя NaN)."""
    array = np.asarray(values, dtype="float64")
    return float(np.nanmean(array))

def median_seconds(func, values):
    """Замеряет время выполнения функции 3 раза и возвращает медиану."""
    func(values) # Первый прогон (разогрев) не учитывается
    marks = []
    for _ in range(3):
        started = time.perf_counter()
        func(values)
        marks.append(time.perf_counter() - started)
    marks.sort()
    return marks[1] # Возвращаем медиану (второе значение из трех)
