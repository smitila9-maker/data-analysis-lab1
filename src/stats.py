import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
import seaborn as sns
import matplotlib.pyplot as plt


def bootstrap_median(values, n_boot=2000, seed=42):
    """
    Считает 95% доверительный интервал для медианы с помощью бутстрапа.
    values: массив чисел (может содержать NaN).
    n_boot: количество повторных выборок (2000 - стандарт).
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype="float64")

    # Убираем пустые значения (NaN)
    values = values[~np.isnan(values)]

    medians = np.empty(n_boot)
    for i in range(n_boot):
        # Берем случайную выборку с возвращением того же размера
        sample = rng.choice(values, size=len(values), replace=True)
        medians[i] = np.median(sample)

    # Возвращаем 2.5 и 97.5 перцентили
    return np.percentile(medians, [2.5, 97.5])


def test_hypothesis_1(
    df,
    group_a,
    group_b,
    group_col="typecode",
    target_col="altitude_1",
):
    """Гипотеза 1: тест Манна-Уитни для двух групп."""
    a = df.loc[df[group_col] == group_a, target_col].dropna()
    b = df.loc[df[group_col] == group_b, target_col].dropna()

    _, p_val = mannwhitneyu(a, b, alternative="two-sided")
    effect = float(a.median() - b.median())
    return float(p_val), effect


def test_hypothesis_2(
    df,
    categories,
    group_col="typecode",
    target_col="altitude_2",
):
    """Гипотеза 2: тест Краскела-Уоллиса и попарные тесты Манна-Уитни с поправкой Холма."""
    groups = {
        name: df.loc[df[group_col] == name, target_col].dropna() for name in categories
    }
    _, p_kruskal = kruskal(*groups.values())

    names = list(groups.keys())
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    raw_p = [
        mannwhitneyu(groups[a], groups[b], alternative="two-sided").pvalue
        for a, b in pairs
    ]

    reject, _, _, _ = multipletests(raw_p, method="holm")
    significant_pairs = int(np.sum(reject))
    return float(p_kruskal), significant_pairs


def test_hypothesis_3(
    df,
    date_threshold,
    threshold_value=1000.0,
    target_col="altitude_1",
    date_col="day",
):
    """Гипотеза 3: z-тест для двух долей (до и после выбранной даты)."""
    ts = pd.Timestamp(date_threshold)
    temp = df.dropna(subset=[target_col, date_col]).copy()
    day_series = pd.to_datetime(temp[date_col])

    if day_series.dt.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize(day_series.dt.tz)
    elif day_series.dt.tz is None and ts.tzinfo is not None:
        ts = ts.tz_convert(None)

    is_high = (temp[target_col] > threshold_value).astype(int)
    before = is_high[day_series < ts]
    after = is_high[day_series >= ts]

    count = [int(after.sum()), int(before.sum())]
    nobs = [int(len(after)), int(len(before))]

    _, p_val = proportions_ztest(count, nobs, alternative="two-sided")
    effect = float(after.mean() - before.mean())
    return float(p_val), effect


def plot_correlation_difference(df, numeric_cols):
    """Строит матрицы Пирсона, Спирмена и их разность."""
    # Считаем корреляции
    pearson = df[numeric_cols].corr(method="pearson")
    spearman = df[numeric_cols].corr(method="spearman")

    # Разность матриц
    diff = (pearson - spearman).abs()

    # Находим пару с наибольшим расхождением (исключая диагональ)
    # Убираем диагональ (там всегда 0)
    diff_no_diag = diff.where(~np.eye(diff.shape[0], dtype=bool))
    max_diff_idx = np.unravel_index(np.nanargmax(diff_no_diag.values), diff.shape)

    col1 = numeric_cols[max_diff_idx[0]]
    col2 = numeric_cols[max_diff_idx[1]]

    print(f"Наибольшее расхождение: {col1} и {col2}")
    print(
        f"Пирсон: {pearson.loc[col1, col2]:.3f}, Спирмен: {spearman.loc[col1, col2]:.3f}"
    )
    print(f"Разница: {diff.loc[col1, col2]:.3f}")

    # Рисуем тепловую карту разности
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(diff, annot=True, cmap="Reds", ax=ax, fmt=".2f")
    ax.set_title("Разница (|Пирсон - Спирмен|)")
    plt.tight_layout()
    plt.savefig("../plots/08_corr_diff.png", dpi=150)
    plt.show()

    return pearson, spearman, diff, (col1, col2)


def plot_simpson(df, flag_col, target_col, group_col):
    """Строит график для проверки парадокса Симпсона."""
    # Общая разность (медианы)
    overall = df.groupby(flag_col)[target_col].median()
    overall_diff = overall[True] - overall[False]

    # Разность внутри групп
    inside = df.groupby([group_col, flag_col])[target_col].median().unstack()

    # --- ИСПРАВЛЕНИЕ: переименовываем столбцы, чтобы избежать KeyError с булевыми значениями ---
    inside = inside.rename(columns={True: "High", False: "Low"})

    inside["diff"] = inside["High"] - inside["Low"]

    print(f"Общая разность медиан: {overall_diff:.2f}")
    print(f"Разности внутри групп:")
    print(inside["diff"].sort_values().head(10))  # Выведем 10 самых больших различий

    # Визуализация (для простоты - первые 5 групп с наибольшей разницей)
    top_groups = inside["diff"].abs().nlargest(5).index
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top_groups))
    width = 0.35

    # Теперь используем новые имена столбцов 'High' и 'Low'
    ax.bar(
        x - width / 2,
        inside.loc[top_groups, "High"],
        width,
        label="Высокая высота (>1000)",
        color="orange",
    )
    ax.bar(
        x + width / 2,
        inside.loc[top_groups, "Low"],
        width,
        label="Низкая высота (<1000)",
        color="blue",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(top_groups, rotation=45)
    ax.set_ylabel("Медиана высоты (altitude_2)")
    ax.set_title("Проверка парадокса Симпсона по типам самолетов")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../plots/09_simpson.png", dpi=150)
    plt.show()

    return overall_diff, inside
