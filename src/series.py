import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


def build_series(df, date_col, freq="D"):
    """Собирает временной ряд по частоте (D - день, W - неделя, M - месяц)."""
    # Считаем количество записей в каждом периоде
    series = df.set_index(date_col).resample(freq).size()
    return series.rename("records")


def plot_series_smoothing(series):
    """Строит график: исходный ряд, скользящее среднее и экспоненциальное сглаживание."""
    smooth = pd.DataFrame(
        {
            "Исходный ряд": series,
            "Скользящее среднее (7)": series.rolling(7, center=True).mean(),
            "Экспоненциальное (alpha=0.3)": series.ewm(alpha=0.3).mean(),
        }
    )

    fig, ax = plt.subplots(figsize=(15, 6))
    smooth.plot(ax=ax, linewidth=1)
    ax.set_title("Временной ряд: исходный, скользящее среднее и EWM")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Число рейсов")
    plt.tight_layout()
    plt.savefig("../plots/10_series_smoothing.png", dpi=150)
    plt.show()
    return smooth


def decompose_series(series, period=7):
    """Декомпозиция ряда (тренд, сезонность, остаток)."""
    # Если данных меньше двух периодов, декомпозиция не сработает
    if len(series) < 2 * period:
        print("Недостаточно данных для декомпозиции!")
        return None

    parts = seasonal_decompose(series, model="additive", period=period)

    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    parts.observed.plot(ax=axes[0], title="Исходный ряд")
    parts.trend.plot(ax=axes[1], title="Тренд")
    parts.seasonal.plot(ax=axes[2], title="Сезонность")
    parts.resid.plot(ax=axes[3], title="Остаток")
    plt.tight_layout()
    plt.savefig("../plots/11_series_decompose.png", dpi=150)
    plt.show()

    return parts


def find_outliers_and_changepoints(parts, window=7):
    """Ищет выбросы по остатку и кандидатов в точки перелома."""
    resid = parts.resid.dropna()

    # 1. Выбросы по остатку (робастное правило 3 сигм)
    spread = 1.4826 * (resid - resid.median()).abs().median()
    outliers = resid[(resid - resid.median()).abs() > 3 * spread]

    print(f"Найдено выбросов по остатку: {len(outliers)}")
    if len(outliers) > 0:
        print(f"Периоды выбросов: {outliers.index.tolist()[:5]}...")

    # 2. Точки перелома (сравнение среднего до и после)
    changepoints = []
    for i in range(window, len(parts.trend) - window):
        before = parts.trend.iloc[i - window : i].mean()
        after = parts.trend.iloc[i : i + window].mean()
        diff = abs(after - before)
        changepoints.append((parts.trend.index[i], diff))

    # Берем топ-3 точки с наибольшей разностью
    changepoints.sort(key=lambda x: x[1], reverse=True)
    print("\nКандидаты в точки перелома:")
    for date, diff in changepoints[:3]:
        print(f"  {date}: разность средних = {diff:.2f}")

    return outliers, changepoints


def plot_autocorrelation(series):
    """Строит график автокорреляции и находит лаг с максимальным значением."""
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(series, lags=14, ax=ax)
    ax.set_title("Автокорреляция временного ряда")
    ax.set_xlabel("Лаг (дни)")
    plt.tight_layout()
    plt.savefig("../plots/12_acf.png", dpi=150)
    plt.show()

    # Находим лаг с наибольшей автокорреляцией (кроме нулевого)
    acf_values = [series.autocorr(lag=i) for i in range(1, 15)]
    best_lag = np.argmax(acf_values) + 1
    best_val = max(acf_values)
    print(f"Лаг с наибольшей автокорреляцией: {best_lag} (значение: {best_val:.3f})")
    return best_lag, best_val


def naive_forecasts(series, h, season, k=4):
    """Считает три наивных прогноза и возвращает ошибки."""
    train, tail = series[:-h], series[-h:]

    # 1. Последнее значение
    last = np.repeat(train.iloc[-1], h)
    # 2. Сезонный прогноз (берем значения того же периода из прошлого цикла)
    seasonal = train.iloc[-season:].to_numpy()[:h]
    # 3. Среднее за последние k периодов
    mean_k = np.repeat(train.iloc[-k:].mean(), h)

    preds = {
        "Последнее значение": last,
        "Сезонный наивный прогноз": seasonal,
        "Среднее за k периодов": mean_k,
    }

    results = {}
    for name, pred in preds.items():
        mae_val = np.mean(np.abs(tail.values - pred))
        # MAPE считаем только если нет нулей в хвосте
        if (tail.values != 0).all():
            mape_val = np.mean(np.abs((tail.values - pred) / tail.values)) * 100
        else:
            mape_val = np.nan
        results[name] = (mae_val, mape_val)

    return tail, preds, results
