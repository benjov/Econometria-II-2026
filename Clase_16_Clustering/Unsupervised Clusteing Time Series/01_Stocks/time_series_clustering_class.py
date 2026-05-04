
"""
time_series_clustering_class.py
--------------------------------
Un módulo didáctico para enseñar clustering de series de tiempo con diferentes enfoques:

1) Basado en características ("feature-based"):
   - Extrae estadísticas (tendencia, autocorrelaciones, espectro, etc.) y aplica K-Means.
2) Basado en forma con distancias elásticas:
   - DTW + clustering jerárquico (y alternativa con clustering espectral sobre una matriz de afinidad).
3) Basado en modelos (parsimonioso):
   - Ajuste AR(p) por mínimos cuadrados y clustering en el espacio de parámetros.
4) Representación simbólica:
   - SAX (PAA + discretización gaussiana) y K-Means sobre bolsa de símbolos.

Incluye utilidades de evaluación (silhouette con métrica euclídea o con distancia precomputada),
visualizaciones, y generación de datos sintéticos con “ground truth”.

Requisitos: numpy, pandas, scipy, scikit-learn, matplotlib (no seaborn).
(NO hace falta tslearn).

Autor: preparado para una clase de Benjamin Oliva.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import numpy as np
import pandas as pd
from scipy import signal
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from pathlib import Path
import matplotlib.pyplot as plt


# ==============================
# Utilidades básicas
# ==============================

def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s < eps:
        return np.zeros_like(x)
    return (x - m) / (s + eps)


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    c = np.convolve(x, np.ones(w), 'valid') / w
    pad_left = w // 2
    pad_right = len(x) - len(c) - pad_left
    return np.pad(c, (pad_left, pad_right), mode='edge')


# ==============================
# Datos sintéticos
# ==============================

@dataclass
class SyntheticConfig:
    n_series: int = 90
    length: int = 120
    n_clusters: int = 3
    noise: float = 0.4
    seed: int = 42


def _cluster_wave(t, freq, phase=0.0, trend=0.0):
    return np.sin(2 * np.pi * freq * t + phase) + trend * t


def make_synthetic_data(cfg: SyntheticConfig = SyntheticConfig()) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera n_series curvas de longitud 'length' pertenecientes a n_clusters grupos con patrones distintos.
    Devuelve:
        X : array (n_series, length)
        y : etiquetas verdaderas (n_series,)
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_series
    L = cfg.length
    k = cfg.n_clusters

    t = np.linspace(0, 1, L)
    X = np.zeros((n, L))
    y = np.zeros(n, dtype=int)

    per_cluster = n // k
    remainder = n - per_cluster * k

    idx = 0

    # Cluster 0: senoidales (distintas fases/frecuencias)
    for i in range(per_cluster + (1 if remainder > 0 else 0)):
        freq = rng.uniform(1.0, 2.0)
        phase = rng.uniform(0, 2*np.pi)
        series = _cluster_wave(t, freq=freq, phase=phase) + rng.normal(scale=cfg.noise, size=L)
        X[idx] = zscore(series)
        y[idx] = 0
        idx += 1

    # Cluster 1: tendencia + estacionalidad suave (tipo "ramps + seasonal")
    for i in range(per_cluster + (1 if remainder > 1 else 0)):
        trend = rng.uniform(0.5, 1.0)
        season = 0.8 * np.sin(2 * np.pi * 1.2 * t + rng.uniform(0, 2*np.pi))
        series = trend*(t - 0.5) + season + rng.normal(scale=cfg.noise, size=L)
        X[idx] = zscore(series)
        y[idx] = 1
        idx += 1

    # Cluster 2: bursts/impulsos + ruido
    for i in range(n - idx):
        base = rng.normal(0, 0.5, size=L)
        for _ in range(rng.integers(2, 5)):
            center = rng.integers(10, L-10)
            width = rng.integers(3, 10)
            amp = rng.uniform(1.0, 2.0)
            burst = amp * signal.windows.gaussian(L, std=width)
            burst = np.roll(burst, center - L//2)
            base += burst
        X[idx] = zscore(base)
        y[idx] = 2
        idx += 1

    return X, y


# ==============================
# Extracción de características
# ==============================

def acf_values(x: np.ndarray, max_lag: int = 10) -> np.ndarray:
    x = x - np.mean(x)
    autocov = np.correlate(x, x, mode='full')
    autocov = autocov[autocov.size // 2:]
    denom = autocov[0] if autocov[0] != 0 else 1.0
    acf = autocov[:max_lag + 1] / denom
    return acf[1:]  # sin lag 0


def linear_trend_slope(x: np.ndarray) -> float:
    n = len(x)
    t = np.arange(n)
    A = np.vstack([t, np.ones(n)]).T
    beta, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
    slope = beta[0]
    return slope


def top_fft_powers(x: np.ndarray, k: int = 5) -> np.ndarray:
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))**2
    # Ignorar el componente DC (índice 0)
    spec = spec[1:]
    if spec.size == 0:
        return np.zeros(k)
    idx = np.argsort(spec)[::-1][:k]
    vals = spec[idx]
    # Si hay menos de k valores, completar con ceros
    if len(vals) < k:
        vals = np.pad(vals, (0, k-len(vals)))
    return vals


def ar_features(x: np.ndarray, p: int = 3) -> np.ndarray:
    """
    Ajusta AR(p) por mínimos cuadrados ordinarios y retorna los coeficientes + std resid.
    """
    x = np.asarray(x)
    n = len(x)
    if n <= p:
        return np.zeros(p + 1)
    Y = x[p:]
    X = np.column_stack([x[p-i-1:n-i-1] for i in range(p)])
    coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef
    sigma = np.std(resid) if resid.size else 0.0
    return np.hstack([coef, sigma])


def sax_breakpoints(alpha: int) -> np.ndarray:
    """
    Umbrales de la N(0,1) para discretización (alfabeto size = alpha).
    """
    from scipy.stats import norm
    qs = np.linspace(0, 1, alpha + 1)[1:-1]
    return norm.ppf(qs)


def sax_paa_vector(x: np.ndarray, n_segments: int = 12, alpha: int = 7) -> np.ndarray:
    """
    SAX simple: z-normaliza -> PAA -> discretiza con puntos de corte gaussianos ->
    bolsa de símbolos (histograma de long alpha).
    """
    x = zscore(np.asarray(x))
    L = len(x)
    n_segments = max(1, min(n_segments, L))
    seg_len = int(np.floor(L / n_segments))
    if seg_len < 1:
        seg_len = 1
        n_segments = L
    means = []
    for i in range(n_segments):
        start = i * seg_len
        end = L if i == n_segments - 1 else (i + 1) * seg_len
        means.append(np.mean(x[start:end]))
    means = np.asarray(means)

    bps = sax_breakpoints(alpha)
    # map to symbols 0..alpha-1
    sym = np.digitize(means, bps)
    hist = np.bincount(sym, minlength=alpha).astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def extract_feature_matrix(X: np.ndarray, p_ar: int = 3, k_fft: int = 5,
                           n_segments_sax: int = 12, sax_alpha: int = 7) -> pd.DataFrame:
    rows = []
    for i in range(X.shape[0]):
        x = X[i]
        feats = {}
        feats['mean'] = float(np.mean(x))
        feats['std'] = float(np.std(x))
        feats['slope'] = float(linear_trend_slope(x))
        acf = acf_values(x, max_lag=4)
        for j, v in enumerate(acf, 1):
            feats[f'acf{j}'] = float(v)
        fftv = top_fft_powers(x, k=k_fft)
        for j, v in enumerate(fftv, 1):
            feats[f'fft_power_{j}'] = float(v)
        ar = ar_features(x, p=p_ar)
        for j, v in enumerate(ar[:-1], 1):
            feats[f'ar{j}'] = float(v)
        feats['ar_sigma'] = float(ar[-1])
        sax = sax_paa_vector(x, n_segments=n_segments_sax, alpha=sax_alpha)
        for j, v in enumerate(sax, 1):
            feats[f'sax_{j}'] = float(v)
        rows.append(feats)
    df = pd.DataFrame(rows)
    return df


# ==============================
# DTW y matrices de distancia
# ==============================

def dtw_distance(a: np.ndarray, b: np.ndarray, window: Optional[int] = None) -> float:
    """
    DTW clásico O(n*m) con ventana opcional de Sakoe-Chiba (en muestras).
    """
    n, m = len(a), len(b)
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))
    inf = 1e18
    D = np.full((n + 1, m + 1), inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        ai = a[i - 1]
        for j in range(j_start, j_end + 1):
            cost = (ai - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def pairwise_dtw_matrix(X: np.ndarray, window: Optional[int] = None) -> np.ndarray:
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(X[i], X[j], window=window)
            D[i, j] = D[j, i] = d
    return D


def distance_to_affinity(D: np.ndarray) -> np.ndarray:
    """
    Convierte una matriz de distancias en una matriz de afinidad con kernel RBF usando
    sigma = mediana(D).
    """
    D = np.asarray(D, dtype=float)
    sigma = np.median(D[D > 0]) if np.any(D > 0) else 1.0
    if sigma <= 0:
        sigma = 1.0
    A = np.exp(- (D ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(A, 0.0)
    return A


# ==============================
# Clustering y evaluación
# ==============================

@dataclass
class ClusterResult:
    labels: np.ndarray
    silhouette: float
    model: object


def cluster_feature_kmeans(features: pd.DataFrame, n_clusters: int, random_state: int = 0) -> Tuple[ClusterResult, np.ndarray, PCA]:
    scaler = StandardScaler()
    Z = scaler.fit_transform(features.values)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(Z)
    sil = silhouette_score(Z, labels, metric='euclidean')
    pca = PCA(n_components=2, random_state=random_state)
    Z2 = pca.fit_transform(Z)
    return ClusterResult(labels=labels, silhouette=float(sil), model=km), Z2, pca


def cluster_dtw_hierarchical(D: np.ndarray, n_clusters: int, linkage: str = 'average') -> ClusterResult:
    # SciPy linkage requiere forma "condensed"
    condensed = squareform(D, checks=False)
    Zlink = hierarchy.linkage(condensed, method=linkage)
    labels = hierarchy.fcluster(Zlink, t=n_clusters, criterion='maxclust') - 1  # 0-index
    sil = silhouette_score(D, labels, metric='precomputed')
    return ClusterResult(labels=labels, silhouette=float(sil), model=Zlink)


def cluster_dtw_spectral(D: np.ndarray, n_clusters: int, random_state: int = 0) -> ClusterResult:
    A = distance_to_affinity(D)
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans',
                            random_state=random_state)
    labels = sc.fit_predict(A)
    sil = silhouette_score(D, labels, metric='precomputed')
    return ClusterResult(labels=labels, silhouette=float(sil), model=sc)


def cluster_model_ar(features_ar: np.ndarray, n_clusters: int, random_state: int = 0) -> ClusterResult:
    scaler = StandardScaler()
    Z = scaler.fit_transform(features_ar)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(Z)
    sil = silhouette_score(Z, labels, metric='euclidean')
    return ClusterResult(labels=labels, silhouette=float(sil), model=km)


# ==============================
# Representantes y visualización
# ==============================

def medoid_indices(D: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
    idxs = {}
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        if len(members) == 1:
            idxs[c] = members[0]
            continue
        subD = D[np.ix_(members, members)]
        sums = subD.sum(axis=1)
        m_idx = members[np.argmin(sums)]
        idxs[c] = m_idx
    return idxs


def plot_feature_space_scatter(Z2: np.ndarray, labels: np.ndarray, title: str, savepath: Optional[str] = None):
    plt.figure(figsize=(6, 5))
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(Z2[idx, 0], Z2[idx, 1], label=f"Cluster {c}", s=20)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    if savepath:
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
    plt.close()


def plot_series_by_cluster(X: np.ndarray, labels: np.ndarray, n_per_cluster: int = 5, savepath: Optional[str] = None):
    plt.figure(figsize=(8, 6))
    n_clusters = len(np.unique(labels))
    L = X.shape[1]
    t = np.arange(L)
    ax = plt.gca()
    for c in range(n_clusters):
        idx = np.where(labels == c)[0][:n_per_cluster]
        for i in idx:
            ax.plot(t, X[i], linewidth=1)
    plt.title("Muestras por cluster (superpuestas)")
    plt.xlabel("t")
    plt.ylabel("valor (z)")
    if savepath:
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
    plt.close()


def plot_medoids(X: np.ndarray, D: np.ndarray, labels: np.ndarray, savepath: Optional[str] = None):
    meds = medoid_indices(D, labels)
    n_clusters = len(meds)
    L = X.shape[1]
    t = np.arange(L)
    plt.figure(figsize=(8, 6))
    for c, i in sorted(meds.items()):
        plt.plot(t, X[i], linewidth=2, label=f"Medoide {c}")
    plt.title("Series medoides por cluster (DTW)")
    plt.xlabel("t")
    plt.ylabel("valor (z)")
    plt.legend()
    if savepath:
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
    plt.close()


def plot_dendrogram(D: np.ndarray, title: str = "Dendrograma (DTW)", savepath: Optional[str] = None):
    condensed = squareform(D, checks=False)
    Zlink = hierarchy.linkage(condensed, method='average')
    plt.figure(figsize=(8, 4))
    hierarchy.dendrogram(Zlink, no_labels=True, color_threshold=None)
    plt.title(title)
    if savepath:
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
    plt.close()


def plot_affinity(A: np.ndarray, title: str = "Afinidad (DTW->RBF)", savepath: Optional[str] = None):
    plt.figure(figsize=(6, 5))
    plt.imshow(A, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    if savepath:
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
    plt.close()


# ==============================
# Pipeline de demostración
# ==============================

@dataclass
class DemoConfig:
    n_series: int = 60
    length: int = 100
    n_clusters: int = 3
    window: Optional[int] = 10  # ventana de DTW (Sakoe-Chiba), None -> sin restricción
    random_state: int = 0
    seed: int = 7


def run_demo(outputs_dir: str = "demo_outputs", cfg: DemoConfig = DemoConfig()) -> Dict[str, float]:
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

    # 1) Datos
    X, y_true = make_synthetic_data(SyntheticConfig(n_series=cfg.n_series, length=cfg.length,
                                                    n_clusters=cfg.n_clusters, seed=cfg.seed))
    dfX = pd.DataFrame(X)
    dfX.to_csv("sample_ts.csv", index=False)

    # 2) Feature-based (mezcla de estadísticas + AR + SAX)
    feats = extract_feature_matrix(X, p_ar=3, k_fft=5, n_segments_sax=12, sax_alpha=7)
    res_feat, Z2, _ = cluster_feature_kmeans(feats, n_clusters=cfg.n_clusters, random_state=cfg.random_state)
    plot_feature_space_scatter(Z2, res_feat.labels, title=f"Feature-based KMeans (sil={res_feat.silhouette:.2f})",
                               savepath=f"{outputs_dir}/feature_space_scatter.png")
    plot_series_by_cluster(X, res_feat.labels, savepath=f"{outputs_dir}/series_by_cluster.png")

    # 3) DTW
    D = pairwise_dtw_matrix(X, window=cfg.window)
    plot_dendrogram(D, savepath=f"{outputs_dir}/dtw_dendrogram.png")

    # 3a) Jerárquico
    res_hier = cluster_dtw_hierarchical(D, n_clusters=cfg.n_clusters, linkage='average')
    plot_medoids(X, D, res_hier.labels, savepath=f"{outputs_dir}/cluster_medoids.png")

    # 3b) Espectral sobre afinidad
    A = distance_to_affinity(D)
    plot_affinity(A, savepath=f"{outputs_dir}/affinity_spectral.png")
    res_spec = cluster_dtw_spectral(D, n_clusters=cfg.n_clusters, random_state=cfg.random_state)

    # 4) "Model-based" simple: parámetros AR(p) y KMeans
    #    (Ya vienen dentro de feats: columnas ar1..arp + ar_sigma)
    ar_cols = [c for c in feats.columns if c.startswith('ar')]
    res_ar = cluster_model_ar(feats[ar_cols].values, n_clusters=cfg.n_clusters, random_state=cfg.random_state)

    return {
        "silhouette_feature_kmeans": res_feat.silhouette,
        "silhouette_dtw_hierarchical": res_hier.silhouette,
        "silhouette_dtw_spectral": res_spec.silhouette,
        "silhouette_ar_model": res_ar.silhouette,
    }


# ==============================
# Si se ejecuta como script
# ==============================

if __name__ == "__main__":
    cfg = DemoConfig()
    metrics = run_demo(outputs_dir="demo_outputs", cfg=cfg)
    print("Resultados demo (silhouette por técnica):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
