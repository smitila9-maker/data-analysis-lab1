import numpy as np
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
