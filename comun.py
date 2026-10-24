"""Piezas estadísticas comunes a los scripts del análisis: una sola semilla para los bootstrap y una sola definición de la
fiabilidad, para que la misma magnitud dé exactamente el mismo intervalo en todas las tablas del manuscrito."""
import numpy as np

SEMILLA_BOOT, N_BOOT = 20261014, 2000


def ic(b):
    """Intervalo percentil del 95 %."""
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def medidas_replicas(x, y, todas):
    """x, y: ln D de las dos corridas de cada sistema replicado; todas: ln D de todas las corridas cortas.
    Ruido de una corrida, fiabilidad con la definición del texto (1 − σ_run²/σ_tot²) e ICC(1) dentro del subconjunto."""
    sd_run = (x - y).std(ddof=1) / np.sqrt(2)
    msb = 2 * np.var((x + y) / 2, ddof=1); msw = np.sum((x - y) ** 2) / (2 * len(x))    # ANOVA de una vía con 2 réplicas
    return {"sd_run": sd_run, "fiabilidad": 1 - sd_run ** 2 / todas.std(ddof=1) ** 2, "icc1_subconjunto": (msb - msw) / (msb + msw)}


def fiabilidad_bootstrap(x, y, todas, n_boot=N_BOOT, semilla=SEMILLA_BOOT):
    """Medidas de réplicas con intervalos bootstrap por sistemas (pares) y por corridas (dispersión total), remuestreados
    con reemplazo. x, y: pd.Series de ln D de las dos corridas, indexadas por sistema. El orden se fija aquí (por sistema;
    todas, ordenadas) y el generador es propio de la llamada, así que cualquier script obtiene el mismo intervalo."""
    x = x.sort_index(); y = y.loc[x.index]; xv, yv = x.to_numpy(float), y.to_numpy(float)
    t = np.sort(np.asarray(todas, float))
    rng = np.random.default_rng(semilla)
    m = medidas_replicas(xv, yv, t); boot = {k: [] for k in m}
    for _ in range(n_boot):
        i = rng.integers(0, len(xv), len(xv)); j = rng.integers(0, len(t), len(t))
        for k, v in medidas_replicas(xv[i], yv[i], t[j]).items(): boot[k].append(v)
    out = {k: float(v) for k, v in m.items()}
    out.update({f"{k}_ic95": ic(boot[k]) for k in m})
    return out


def top_k(valores, k):
    """Posiciones de los k mayores valores. Orden estable: en un remuestreo con reemplazo, las copias de un mismo sistema
    (valores idénticos) se ordenan igual en la etiqueta corta y en la referencia, sin crear fallos artificiales."""
    return np.argsort(-np.asarray(valores, float), kind="stable")[:k]


def recall_top(corto, referencia, frac):
    """Recall del top-k de la referencia al elegir los k mejores por la corrida corta (k = fracción del total, mínimo 1).
    Trabaja por posiciones, así que admite muestras bootstrap con sistemas repetidos."""
    k = max(1, round(frac * len(corto)))
    aciertos = len(set(top_k(referencia, k)) & set(top_k(corto, k)))
    return {"k": k, "aciertos": aciertos, "recall": aciertos / k, "azar": k / len(corto)}
