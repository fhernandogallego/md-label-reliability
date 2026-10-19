"""Comprobación con datos reales de la asignación adaptativa (complementa asignacion.py, que usa el modelo de ruido).

Datos: los 94 sistemas de cribado con dos corridas de 5 ns. Por las carpetas de HTP-MD, la primera (A, carpeta
poly_9_nion_50_conc_1.5) es la corrida original del cribado y la segunda (B, poly_5ns_new_config_rand_100) parte de una
estructura nueva; 92 de ellos tienen además una corrida de 50 ns, que parte de la misma estructura que A.

1) Calibración del criterio. La asignación adaptativa decide a quién repetir con la probabilidad posterior de estar por
   encima del corte. Aquí, con una corrida X de cada sistema, el modelo predice P(Y > c | X) para la otra corrida Y (ruido
   predictivo = incertidumbre posterior + ruido de una corrida), con c el corte del top-k fijado con X. Se compara con lo
   observado (Brier frente a la tasa base, calibración por tramos), en los dos sentidos (A→B y B→A). Los parámetros del ruido
   se estiman sin el sistema evaluado (dejando uno fuera), para que la predicción sea fuera de muestra.
2) Minicampaña real a igual presupuesto. Todos reciben A; la mitad recibe además B, elegida por la frontera (mayor p·(1−p))
   o al azar (control: misma cantidad de segundas corridas); además A sola y A+B para todos. Selección por media posterior;
   evaluación: recall del top-k según la corrida de 50 ns (independiente de B). Intervalos bootstrap por sistemas.

Lee resultados/tabla_por_trayectoria.csv, resultados/incertidumbre.json y resultados/techo.json (no necesita data/).
Uso: python validacion_asignacion.py → resultados/validacion_asignacion.json, paper/numeros_validacion.tex
"""
import json
import numpy as np
import pandas as pd
from scipy import optimize, stats

N_BOOT, N_AZAR, LOTE, SEMILLA = 2000, 2000, 5, 20261016
TOPS = (0.10, 0.20)
T = pd.read_csv("resultados/tabla_por_trayectoria.csv")
cortas = T[T.ns == 5].sort_values("id")
rep = cortas.groupby("sistema").filter(lambda g: len(g) == 2)
A = rep[rep.carpeta.str.startswith("poly_9_nion")].set_index("sistema").D_extremos
B = rep[rep.carpeta.str.contains("new_config")].set_index("sistema").D_extremos
sis = sorted(set(A.index) & set(B.index)); xa, xb = np.log(A[sis].values), np.log(B[sis].values)
L = T[T.ns == 50].set_index("sistema").D_medio
S_TOT = float(np.log(cortas.D_extremos).std(ddof=1))
rng = np.random.default_rng(SEMILLA)
R = {"n_replicados": len(sis), "n_boot": N_BOOT, "semilla": SEMILLA}


def parametros(x, y):
    """σ_run medio y por mitades (lenta/rápida) a partir de pares de réplicas; σ_sis de la dispersión total."""
    d = x - y; nivel = (x + y) / 2; mitad = nivel < np.median(nivel)
    s_run = d.std(ddof=1) / np.sqrt(2); lento = d[mitad].std(ddof=1) / np.sqrt(2); rapido = d[~mitad].std(ddof=1) / np.sqrt(2)
    s_sis = np.sqrt(max(S_TOT ** 2 - s_run ** 2, 1e-6))
    return {"s_sis": s_sis, "a": (lento + rapido) / 2, "b": (rapido - lento) / (2 * 0.798), "s_run": s_run}


def sigma(m, P):
    return np.clip(P["a"] + P["b"] * m / P["s_sis"], 0.08, None)


def posterior(suma, n, P, centro):
    """Posterior de μ con prior N(centro, σ_sis²) y ruido σ(media observada)."""
    media = suma / n; s = sigma(media - centro, P)
    prec = 1 / P["s_sis"] ** 2 + n / s ** 2
    return (centro / P["s_sis"] ** 2 + n * media / s ** 2) / prec, 1 / np.sqrt(prec), s


# ---------- 1) Calibración, fuera de muestra (dejando uno fuera) ----------
centro = float(np.log(cortas.D_extremos).mean())
R["calibracion"] = {}
for top in TOPS:
    k = max(1, round(top * len(sis))); pred, obs = [], []
    for x, y in ((xa, xb), (xb, xa)):
        c = np.sort(x)[-k]                                                  # corte del top-k fijado con X
        for i in range(len(sis)):
            P = parametros(np.delete(xa, i), np.delete(xb, i))
            m, s_post, s_obs = posterior(x[i:i + 1], np.ones(1), P, centro)
            pred.append(float(1 - stats.norm.cdf((c - m[0]) / np.sqrt(s_post[0] ** 2 + s_obs[0] ** 2))))
            obs.append(float(y[i] >= c))
    pred, obs = np.array(pred), np.array(obs)
    brier = float(np.mean((pred - obs) ** 2)); base = float(np.mean((obs.mean() - obs) ** 2))

    def pendiente(p, o):
        """Pendiente de calibración: regresión logística de lo observado sobre logit(predicho); 1 = bien calibrado, < 1 = sobreconfiado.
        Penalización L2 mínima para que el ajuste no diverja si un remuestreo separa los datos."""
        z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        f = lambda w: np.sum(np.logaddexp(0, w[0] + w[1] * z) - o * (w[0] + w[1] * z)) + 1e-2 * w @ w
        return float(optimize.minimize(f, np.array([0.0, 1.0]), method="BFGS").x[1])
    sistemas = np.tile(np.arange(len(sis)), 2)                               # remuestreo por sistema (sus dos predicciones juntas)
    bp = []
    for _ in range(N_BOOT):
        e_ = rng.integers(0, len(sis), len(sis)); idx = np.concatenate([np.where(sistemas == j)[0] for j in e_])
        bp.append(pendiente(pred[idx], obs[idx]))
    tramos = [(0, .1), (.1, .5), (.5, 1.01)]
    R["calibracion"][str(top)] = {
        "n_predicciones": len(pred), "brier": brier, "brier_tasa_base": base, "mejora_brier_pct": 100 * (1 - brier / base),
        "pendiente_calibracion": pendiente(pred, obs), "pendiente_ic95": [float(np.quantile(bp, .025)), float(np.quantile(bp, .975))],
        "tramos": [{"pred_desde": a, "pred_hasta": min(b, 1), "n": int(((pred >= a) & (pred < b)).sum()),
                    "pred_media": float(pred[(pred >= a) & (pred < b)].mean()) if ((pred >= a) & (pred < b)).any() else None,
                    "obs_frecuencia": float(obs[(pred >= a) & (pred < b)].mean()) if ((pred >= a) & (pred < b)).any() else None}
                   for a, b in tramos]}

# ---------- 2) Minicampaña real a igual presupuesto ----------
con_larga = [i for i, s in enumerate(sis) if s in L.index]
ya, yb, ref = xa[con_larga], xb[con_larga], np.log(L[[sis[i] for i in con_larga]].values)
n = len(ref); P = parametros(xa, xb); extra = n // 2
R["campana"] = {"n": n, "segundas_corridas": extra}


def seleccion(usa_b, k):
    suma = ya + np.where(usa_b, yb, 0); nn = 1 + usa_b.astype(float)
    m, _, _ = posterior(suma, nn, P, centro)
    return np.argsort(-m)[:k]


def frontera(k):
    """Segundas corridas por lotes a los de mayor p·(1−p) respecto al corte del top-k, recalculado tras cada lote."""
    usa_b = np.zeros(n, bool)
    while usa_b.sum() < extra:
        suma = ya + np.where(usa_b, yb, 0); nn = 1 + usa_b.astype(float)
        m, s_post, _ = posterior(suma, nn, P, centro); c = np.sort(m)[-k]
        p = stats.norm.cdf((m - c) / s_post); duda = np.where(usa_b, -1, p * (1 - p))
        usa_b[np.argsort(-duda)[:min(LOTE, extra - usa_b.sum())]] = True
    return usa_b


def recall(sel, k, idx=None):
    r = ref if idx is None else ref[idx]
    verdad = set(np.argsort(-r)[:k]); return len(verdad & set(sel)) / k


for top in TOPS:
    k = max(1, round(top * n))
    usa_f = frontera(k)
    e = {"k": k,
         "solo_A": recall(seleccion(np.zeros(n, bool), k), k),
         "frontera": recall(seleccion(usa_f, k), k),
         "azar": float(np.mean([recall(seleccion(np.isin(np.arange(n), rng.choice(n, extra, replace=False)), k), k) for _ in range(N_AZAR)])),
         "A_y_B_todos": recall(seleccion(np.ones(n, bool), k), k)}
    # bootstrap por sistemas (sin duplicados) de la diferencia frontera − azar
    difs = []
    for _ in range(N_BOOT):
        idx = np.unique(rng.integers(0, n, n)); kk = max(1, round(top * len(idx)))
        r_ = ref[idx]; verdad = set(np.argsort(-r_)[:kk])
        def sel_sub(usa):
            suma = ya[idx] + np.where(usa, yb[idx], 0); nn = 1 + usa.astype(float)
            m, _, _ = posterior(suma, nn, P, centro); return set(np.argsort(-m)[:kk])
        usa_az = np.zeros(len(idx), bool); usa_az[rng.choice(len(idx), int(usa_f[idx].sum()), replace=False)] = True
        difs.append((len(verdad & sel_sub(usa_f[idx])) - len(verdad & sel_sub(usa_az))) / kk)
    e["dif_frontera_azar_ic95"] = [float(np.quantile(difs, .025)), float(np.quantile(difs, .975))]
    e["prob_frontera_mejor_o_igual"] = float(np.mean(np.array(difs) >= 0))
    R["campana"][str(top)] = e

json.dump(R, open("resultados/validacion_asignacion.json", "w"), indent=1)

pc = lambda x: f"{100 * x:.0f}"
M = {"valAlNRep": str(R["n_replicados"]), "valAlNCamp": str(n), "valAlExtra": str(extra)}
for top, nt in ((0.10, "Ten"), (0.20, "Twenty")):
    C = R["calibracion"][str(top)]; E = R["campana"][str(top)]
    M[f"valAlBrierGain{nt}"] = f"{C['mejora_brier_pct']:.0f}"
    M[f"valAlCalSlope{nt}"] = f"{C['pendiente_calibracion']:.2f}"; M[f"valAlCalSlopeLo{nt}"], M[f"valAlCalSlopeHi{nt}"] = (f"{v:.2f}" for v in C["pendiente_ic95"])
    M[f"valAlK{nt}"] = str(E["k"])
    for kk_, nk in (("solo_A", "OnlyA"), ("frontera", "Bound"), ("azar", "Random"), ("A_y_B_todos", "Both")):
        M[f"valAl{nk}{nt}"] = pc(E[kk_])
    M[f"valAlDiffLo{nt}"], M[f"valAlDiffHi{nt}"] = (pc(v) for v in E["dif_frontera_azar_ic95"])
with open("paper/numeros_validacion.tex", "w") as fh:
    fh.write("% Generado por validacion_asignacion.py. No editar a mano.\n")
    for kk_, v in M.items(): fh.write(f"\\newcommand{{\\{kk_}}}{{{v}}}\n")

for top in TOPS:
    C = R["calibracion"][str(top)]
    print(f"\ncalibración top {top:.0%}: {C['n_predicciones']} predicciones, Brier {C['brier']:.3f} frente a {C['brier_tasa_base']:.3f} "
          f"(tasa base) → mejora {C['mejora_brier_pct']:.0f} %; pendiente de calibración {C['pendiente_calibracion']:.2f} {[round(v, 2) for v in C['pendiente_ic95']]}")
    for t in C["tramos"]:
        if t["n"]: print(f"   predicho {t['pred_desde']:.1f}–{t['pred_hasta']:.1f}: n={t['n']:3d}, predicho medio {t['pred_media']:.2f}, observado {t['obs_frecuencia']:.2f}")
    E = R["campana"][str(top)]
    print(f"campaña top {top:.0%} (k={E['k']} de {n}, {extra} segundas corridas): solo A {E['solo_A']:.2f} | frontera {E['frontera']:.2f} | "
          f"azar {E['azar']:.2f} | A+B todos {E['A_y_B_todos']:.2f}; frontera − azar IC95 {E['dif_frontera_azar_ic95']}, "
          f"P(frontera ≥ azar) {E['prob_frontera_mejor_o_igual']:.2f}")
