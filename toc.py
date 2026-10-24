"""Gráfico para la tabla de contenidos de JCIM (8,25 cm x 4,45 cm), esquemático pero con las cifras de los resultados.

Izquierda: etiquetas simuladas de 5 ns frente a la propiedad de cada sistema, con el ruido medido (techo √r) y la correlación
del random forest. Derecha: candidatos ordenados por su primera corrida y réplicas extra concentradas cerca del corte del
top 10 % (asignación a la frontera), con el ahorro frente a cuatro réplicas uniformes.
Lee resultados/incertidumbre.json, techo.json, ml_demo.json y asignacion.json.
Uso: python toc.py → paper/figures/toc.pdf y toc.png
"""
import json
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, NARANJA, VERDE, TINTA, TINTA2, REJ = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TINTA2,
                     "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2, "pdf.fonttype": 42,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
U = json.load(open("resultados/incertidumbre.json"))["replicas"]["D_extremos"]
techo = json.load(open("resultados/techo.json"))["D_extremos"]["techo_correlacion"]
rf = json.load(open("resultados/ml_demo.json"))["todas"]["pearson"]
ahorro = json.load(open("resultados/asignacion.json"))["equivalencia"]["D_extremos|normal|0.1|4"]["ahorro"]
r, s_run = U["fiabilidad"], U["sd_run"]; s_sis = np.sqrt(r * s_run ** 2 / (1 - r))
rng = np.random.default_rng(7)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.25 / 2.54, 4.45 / 2.54), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.35})

# --- izquierda: ruido de las etiquetas ---
mu = rng.normal(0, s_sis, 220); lab = mu + rng.normal(0, s_run, mu.size)
a1.scatter(mu, lab, s=2.5, color=AZUL, alpha=0.5, lw=0)
lim = np.array([-1.5, 1.5]); a1.plot(lim, lim, color=TINTA2, lw=0.6, ls=(0, (2, 2)))
a1.set_xlim(lim); a1.set_ylim(lim); a1.set_xticks([]); a1.set_yticks([])
a1.set_xlabel("System property"); a1.set_ylabel("One 5 ns MD label")
a1.text(1.45, -1.42, f"max. correlation ≈ {techo:.2f}\nML model: {rf:.2f}", ha="right", va="bottom", fontsize=5.6, color=TINTA,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
a1.set_title("Short runs are noisy labels", fontsize=6.6, pad=3)

# --- derecha: asignación adaptativa real (frontera), por tandas ---
N, k, extra_total, tanda = 40, 4, 40, 4
mu = rng.normal(0, s_sis, N)
def post(suma, n):
    prec = 1 / s_sis ** 2 + n / s_run ** 2; return (suma / s_run ** 2) / prec, 1 / np.sqrt(prec)
suma = mu + rng.normal(0, s_run, N); n = np.ones(N); orden = np.argsort(-suma)
resto = extra_total
while resto > 0:
    m, sp = post(suma, n); c = np.sort(m)[-k]; p = stats.norm.cdf((m - c) / sp)
    sel = np.argsort(-(p * (1 - p)))[:min(tanda, resto)]
    suma[sel] += mu[sel] + rng.normal(0, s_run, len(sel)); n[sel] += 1; resto -= len(sel)
x = np.arange(N)
a2.bar(x, np.ones(N), color=REJ, width=0.8, lw=0)
a2.bar(x, n[orden] - 1, bottom=1, color=NARANJA, width=0.8, lw=0)
a2.axvline(k - 0.5, color=TINTA, lw=0.6, ls=(0, (2, 2)))
a2.text(k + 0.4, n.max() + 0.9, "top-k cut-off", fontsize=5.4, color=TINTA, va="center")
a2.set_xticks([]); a2.set_ylim(0, n.max() + 1.6); a2.set_yticks([1, int(n.max() // 2), int(n.max())])
a2.set_xlabel("Candidates ranked by first run"); a2.set_ylabel("Runs")
a2.text(N - 0.5, 0.55 * n.max(), f"same recall as\n4 runs each,\n{ahorro:.1f}× less MD", ha="right", va="center", fontsize=5.6, color=NARANJA)
a2.set_title("Re-run only where it matters", fontsize=6.6, pad=3)

fig.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.90)          # tamaño exacto 8,25 x 4,45 cm (sin recorte)
fig.savefig("paper/figures/toc.pdf")
fig.savefig("paper/figures/toc.png", dpi=600)
print(f"toc: techo {techo:.2f}, RF {rf:.2f}, ahorro ×{ahorro:.2f}")
