import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def prepare_data(df, cols_to_scale):
    """Стандартизирует признаки (вычитает среднее, делит на отклонение)."""
    print("Стандартизация признаков...")
    scaler = StandardScaler()
    # Берем только нужные колонки и убираем NaN
    X = df[cols_to_scale].dropna()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X.index


def elbow_silhouette(X_scaled, k_range=range(2, 11)):
    """Считает инерцию и силуэт для разных k (метод локтя)."""
    print("Перебор k от 2 до 10 (это может занять пару минут)...")
    inertia = []
    silhouette = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

        # Силуэт считается долго на больших данных, поэтому берем подвыборку
        # (если данных много)
        if len(X_scaled) > 50000:
            idx = np.random.choice(len(X_scaled), 50000, replace=False)
            score = silhouette_score(X_scaled[idx], kmeans.labels_[idx])
        else:
            score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette.append(score)
        print(f"k={k}: инерция={kmeans.inertia_:.0f}, силуэт={score:.3f}")

    # Строим графики
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_range, inertia, "bo-")
    ax1.set_xlabel("Число кластеров (k)")
    ax1.set_ylabel("Инерция")
    ax1.set_title("Метод локтя")

    ax2.plot(k_range, silhouette, "ro-")
    ax2.set_xlabel("Число кластеров (k)")
    ax2.set_ylabel("Силуэт")
    ax2.set_title("Силуэт")

    plt.tight_layout()
    plt.savefig("../plots/10_clusters_k.png", dpi=150)
    plt.show()

    return inertia, silhouette


def run_dbscan(X_scaled, eps=0.5, min_samples=5):
    """Запускает DBSCAN и возвращает долю шума."""
    print(f"Запуск DBSCAN (eps={eps}, min_samples={min_samples})...")
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_scaled)

    # Считаем долю шума (метка -1)
    noise_share = (labels == -1).sum() / len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"DBSCAN: кластеров = {n_clusters}, доля шума = {noise_share * 100:.2f}%")
    return labels, noise_share
