import numpy as np
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest


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


def test_hypothesis_1(df, group_a, group_b, target_col="altitude_1"):
    """Гипотеза 1: Манна-Уитни для двух групп."""
    a = df[df["origin"] == group_a][target_col].dropna()
    b = df[df["origin"] == group_b][target_col].dropna()

    stat, p = mannwhitneyu(a, b)
    # Размер эффекта - разница медиан
    effect = a.median() - b.median()
    return p, effect


def test_hypothesis_2(df, top_airports, target_col="altitude_1"):
    """Гипотеза 2: Краскела-Уоллиса и попарные тесты."""
    groups = {
        name: df[df["origin"] == name][target_col].dropna() for name in top_airports
    }

    # Общий тест
    stat, p_kruskal = kruskal(*groups.values())

    # Попарные тесты
    names = list(groups)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    p_values = []
    for a, b in pairs:
        _, p = mannwhitneyu(groups[a], groups[b])
        p_values.append(p)

    # Поправка Холма
    reject, adjusted_p, _, _ = multipletests(p_values, method="holm")
    significant_pairs = sum(reject)

    return p_kruskal, significant_pairs


def test_hypothesis_3(df, date_threshold, target_col="altitude_1"):
    """Гипотеза 3: z-тест для двух долей (до и после даты)."""
    import pandas as pd
    from statsmodels.stats.proportion import proportions_ztest

    # Приводим дату к тому же часовому поясу, что и в данных
    date_threshold = pd.Timestamp(date_threshold, tz="UTC")

    df_temp = df.dropna(subset=[target_col, "day"]).copy()
    df_temp["is_high"] = (df_temp[target_col] > 1000).astype(int)

    before = df_temp[df_temp["day"] < date_threshold]["is_high"]
    after = df_temp[df_temp["day"] >= date_threshold]["is_high"]

    count = [before.sum(), after.sum()]
    nobs = [len(before), len(after)]

    stat, p = proportions_ztest(count, nobs)
    effect = after.mean() - before.mean()
    return p, effect
