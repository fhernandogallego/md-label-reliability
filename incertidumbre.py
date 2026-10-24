"""Incertidumbre de los resultados de réplicas y pares, y utilidad de selección de las corridas cortas.

Lee resultados/tabla_por_trayectoria.csv (generada por analizar_curvas.py), así que no necesita data/.
  - Fiabilidad de una corrida de 5 ns: la definición del texto (ruido de las réplicas frente a la dispersión de todas las
    corridas cortas) y el ICC(1) estimado solo dentro del subconjunto replicado, ambos con intervalo bootstrap por sistemas.
  - Escala de error para comparar DOS candidatos: la diferencia de dos corridas tiene desviación √2·σ_run.
  - Selección: recall/precisión del k mejor según la corrida corta frente a la larga, para varios k, con intervalos
    bootstrap por sistemas (con reemplazo; la selección es por posiciones) y la línea base de una selección aleatoria.
  Semilla y fiabilidad comunes en comun.py.
Uso: python incertidumbre.py → resultados/incertidumbre.json y paper/tabla_seleccion.tex
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from comun import N_BOOT, SEMILLA_BOOT as SEMILLA, ic, fiabilidad_bootstrap, recall_top

rng = np.random.default_rng(SEMILLA)
T = pd.read_csv("resultados/tabla_por_trayectoria.csv")
cortas = T[T.ns == 5].sort_values("id")
R = {"n_boot": N_BOOT, "semilla": SEMILLA}

# ---------- Réplicas: ruido, fiabilidad e ICC ----------
rep = cortas.groupby("sistema").filter(lambda g: len(g) == 2).sort_values(["sistema", "id"])
R["replicas"] = {"n_sistemas": int(rep.sistema.nunique())}
for col in ("D_extremos", "D_medio"):
    g = rep.groupby("sistema")[col]                                    # comun.fiabilidad_bootstrap: misma cifra e IC en todo el texto
    R["replicas"][col] = fiabilidad_bootstrap(np.log(g.first()), np.log(g.last()), np.log(cortas[col].values))
    s2 = np.sqrt(2) * R["replicas"][col]["sd_run"]                                  # diferencia entre dos candidatos medidos con una corrida cada uno
    R["replicas"][col].update({"factor_diferencia_1sd": float(np.exp(s2)), "factor_diferencia_95": float(np.exp(1.96 * s2))})

# ---------- Pares corto/largo: correlación y selección ----------
a = cortas.groupby("sistema").agg(D5_extremos=("D_extremos", "first"), D5_medio=("D_medio", "first"), literatura=("literatura", "first"))
b = T[T.ns == 50].set_index("sistema")[["D_medio"]].rename(columns={"D_medio": "D50"})
P = a.join(b, how="inner"); n = len(P)
R["pares"] = {"n": n, "n_cribado": int((~P.literatura).sum()), "n_literatura": int(P.literatura.sum())}


def seleccion(Q, col, frac):
    """Recall del top-k de la corrida larga eligiendo por la corta; por posiciones (admite sistemas repetidos)."""
    return recall_top(Q[col].to_numpy(), Q.D50.to_numpy(), frac)


for col in ("D5_extremos", "D5_medio"):
    e = {"spearman": float(stats.spearmanr(P[col], P.D50)[0])}
    bs = []
    for _ in range(N_BOOT):
        Q = P.iloc[rng.integers(0, n, n)]; bs.append(stats.spearmanr(Q[col], Q.D50)[0])
    e["spearman_ic95"] = ic(bs)
    for sub, Q in (("cribado", P[~P.literatura]), ("literatura", P[P.literatura])):
        e[f"spearman_{sub}"] = float(stats.spearmanr(Q[col], Q.D50)[0])
    for frac in (0.05, 0.10, 0.20):
        s = seleccion(P, col, frac); br = []
        for _ in range(N_BOOT):                                         # bootstrap por sistemas, con reemplazo
            Q = P.iloc[rng.integers(0, n, n)]; br.append(seleccion(Q, col, frac)["recall"])
        s["recall_ic95"] = ic(br)
        e[f"top{round(100 * frac)}"] = s
    R["pares"][col] = e

# selección dentro del cribado: el top de los 133 pares lo forman casi solo polímeros de literatura (al final, para no
# alterar los números aleatorios de lo anterior)
S = P[~P.literatura]; nS = len(S); R["pares"]["cribado"] = {"n": nS}
for frac in (0.05, 0.10, 0.20):
    kk = max(1, round(frac * n)); R["pares"]["cribado"][f"literatura_en_top{round(100 * frac)}_de_todos"] = int(P.loc[P.D50.nlargest(kk).index, "literatura"].sum())
    s = seleccion(S, "D5_extremos", frac); br = []
    for _ in range(N_BOOT):
        Q = S.iloc[rng.integers(0, nS, nS)]; br.append(seleccion(Q, "D5_extremos", frac)["recall"])
    s["recall_ic95"] = ic(br); R["pares"]["cribado"][f"top{round(100 * frac)}"] = s
json.dump(R, open("resultados/incertidumbre.json", "w"), indent=1)

with open("paper/tabla_seleccion.tex", "w") as fh:
    fh.write("% Generado por incertidumbre.py. No editar a mano.\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
             "Estimator, \\SI{5}{\\nano\\second} run & Top & $k$ & Hits & Recall (95\\% CI) & Random \\\\\n\\midrule\n")
    for col, nom in (("D5_extremos", "End-point"), ("D5_medio", "Origin-averaged, lag $T/2$")):
        for j, frac in enumerate((5, 10, 20)):
            s = R["pares"][col][f"top{frac}"]; lo, hi = s["recall_ic95"]
            fh.write(f"{nom if j == 0 else ''} & {frac}\\% & {s['k']} & {s['aciertos']} & {100 * s['recall']:.0f}\\% ({100 * lo:.0f} to {100 * hi:.0f}) & {100 * s['azar']:.0f}\\% \\\\\n")
    for j, frac in enumerate((5, 10, 20)):
        s = R["pares"]["cribado"][f"top{frac}"]; lo, hi = s["recall_ic95"]
        fh.write(f"{'End-point, screening only' if j == 0 else ''} & {frac}\\% & {s['k']} & {s['aciertos']} & {100 * s['recall']:.0f}\\% ({100 * lo:.0f} to {100 * hi:.0f}) & {100 * s['azar']:.0f}\\% \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

print(json.dumps(R, indent=1))
