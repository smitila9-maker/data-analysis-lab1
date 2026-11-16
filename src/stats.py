import numpy as np


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
