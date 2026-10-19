"""Asignación adaptativa de réplicas: ¿cuánto cómputo hace falta para encontrar los mejores candidatos?

Une la prueba de viabilidad de M. (antes exploracion/asignacion.py: modelo posterior, ruido continuo con la movilidad,
successive halving, criterio de frontera p·(1−p), forma empírica) con la de F. (lectura del ruido medido, comprobación de
gaussianidad, top 10 %, presupuesto equivalente, tabla/macros/figura para el manuscrito).

Problema: cribado de los N = 6.137 candidatos con corridas cortas ruidosas; objetivo, los k mejores (mayor difusividad del
Li). Presupuesto medio b corridas por candidato. Estrategias:
  uniforme   b corridas a cada candidato (b entero)
  mitades    successive halving: 1 corrida a todos; después otra a los mejores 4k, 2k, k, y al llegar a k se empieza
             una nueva ronda desde 4k, hasta agotar el presupuesto (sin el reinicio se gastaría todo en los mismos k)
  frontera   1 corrida a todos; después, por tandas del 5 % de N, otra a los candidatos con más dudas de estar en el top-k
             (mayor p·(1−p), p = P(estar por encima del corte) con el posterior normal)
Selección final por la media posterior. Cada réplica parte de una estructura nueva, así que lleva el ruido completo σ_run.

Modelo de ruido (ln D, de techo.py e incertidumbre.py: 94 sistemas con réplicas):
  valor del sistema μ ~ N(0, σ_sis²), σ_sis² = r·σ_run²/(1−r)      (o forma empírica de las cortas, escalada a σ_sis)
  corrida = μ + ε, ε ~ N(0, σ(μ)²), σ(μ) = a + b·μ/σ_sis, ajustada a σ_run en la mitad lenta y en la rápida (±0,798 = media
  de |z| en cada mitad de una normal); variante "frontera_homo" con σ constante para ver si la heterocedasticidad aporta.
Se comprueba que las diferencias entre réplicas reales son gaussianas. Métrica: recall del verdadero top-k.

Lee resultados/techo.json, resultados/incertidumbre.json y resultados/tabla_por_trayectoria.csv (no necesita data/).
Uso: python asignacion.py → resultados/asignacion.json, paper/tabla_asignacion.tex, paper/numeros_asignacion.tex,
     paper/figures/fig8_allocation.pdf
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_MC, N_MC_ROBUSTEZ, TANDA, SEMILLA = 200, 100, 0.05, 20261017
PRESUPUESTOS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
TOPS = [0.01, 0.05, 0.10]
PRINCIPAL = ("D_extremos", "normal")                 # etiqueta de HTP-MD (end-point) y valores normales
AZUL, NARANJA, VERDE, TINTA, TINTA2, REJ = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": REJ, "grid.linewidth": 0.6,
                     "lines.linewidth": 2.0, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})

T = json.load(open("resultados/techo.json")); U = json.load(open("resultados/incertidumbre.json"))["replicas"]
cortas = pd.read_csv("resultados/tabla_por_trayectoria.csv").query("ns == 5").sort_values("id")
N = len(cortas)


def modelo(est):
    p = T[est]; r, sr = U[est]["fiabilidad"], U[est]["sd_run"]
    lento, rapido = p["heterocedasticidad"]["sd_run_lentos"], p["heterocedasticidad"]["sd_run_rapidos"]
    return {"s_sis": float(np.sqrt(r * sr ** 2 / (1 - r))), "a": (lento + rapido) / 2, "b": (rapido - lento) / (2 * 0.798),
            "sd_run": sr, "fiabilidad": r}


def sigma(m, M, homo=False):
    return np.full_like(m, M["sd_run"]) if homo else np.clip(M["a"] + M["b"] * m / M["s_sis"], 0.08, None)


def verdaderos(est, M, forma, rng):
    if forma == "normal": return rng.normal(0, M["s_sis"], N)
    x = np.log(cortas[est].values); x = (x - x.mean()) / x.std()
    return rng.choice(x, N, replace=True) * M["s_sis"]


def posterior(suma, n, M, homo=False):
    """Media y sd posteriores con prior N(0, σ_sis²) y ruido σ evaluado en la media observada."""
    media = suma / np.maximum(n, 1); s = sigma(media, M, homo)
    prec = 1 / M["s_sis"] ** 2 + n / s ** 2
    return (n * media / s ** 2) / prec, 1 / np.sqrt(prec)


def correr(mu, idx, M, rng):
    return mu[idx] + rng.standard_normal(len(idx)) * sigma(mu[idx], M)    # el ruido real siempre depende de la movilidad


def mejores(m, k):
    return np.argpartition(-m, k)[:k]


def uniforme(mu, k, b, M, rng):
    n = np.full(N, int(b)); suma = np.zeros(N)
    for _ in range(int(b)): suma += correr(mu, np.arange(N), M, rng)
    return mejores(posterior(suma, n, M)[0], k)


def mitades(mu, k, b, M, rng):
    n = np.ones(N); suma = correr(mu, np.arange(N), M, rng); resto = int(round((b - 1) * N)); g = 4 * k
    while resto > 0:
        sel = mejores(posterior(suma, n, M)[0], min(g, resto, N - 1))
        suma[sel] += correr(mu, sel, M, rng); n[sel] += 1; resto -= len(sel)
        g = g // 2 if g // 2 >= k else 4 * k                                 # al llegar a k, nueva ronda desde 4k (no repetir siempre los mismos k)
    return mejores(posterior(suma, n, M)[0], k)


def frontera(mu, k, b, M, rng, homo=False):
    n = np.ones(N); suma = correr(mu, np.arange(N), M, rng); resto = int(round((b - 1) * N))
    while resto > 0:
        m, s = posterior(suma, n, M, homo); c = np.partition(m, N - k)[N - k]
        p = stats.norm.cdf((m - c) / s); lote = min(int(TANDA * N), resto)
        sel = np.argpartition(-(p * (1 - p)), lote)[:lote]
        suma[sel] += correr(mu, sel, M, rng); n[sel] += 1; resto -= lote
    return mejores(posterior(suma, n, M, homo)[0], k)


ESTRATEGIAS = {"uniforme": uniforme, "mitades": mitades, "frontera": frontera,
               "frontera_homo": lambda mu, k, b, M, rng: frontera(mu, k, b, M, rng, homo=True)}

# ---------- Comprobación: ¿son gaussianas las diferencias entre réplicas reales? ----------
rep = cortas.groupby("sistema").filter(lambda g: len(g) == 2).sort_values(["sistema", "id"])
d = np.log(rep.groupby("sistema").D_extremos.first().values) - np.log(rep.groupby("sistema").D_extremos.last().values)
z = (d - d.mean()) / d.std(ddof=1)
R = {"N": N, "n_mc": N_MC, "n_mc_robustez": N_MC_ROBUSTEZ, "tanda": TANDA, "semilla": SEMILLA, "principal": list(PRINCIPAL),
     "gaussianidad": {"n": len(d), "curtosis_exceso": float(stats.kurtosis(d)), "shapiro_p": float(stats.shapiro(d).pvalue),
                      "frac_z_mayor_2": float(np.mean(np.abs(z) > 2)), "frac_z_mayor_2_5": float(np.mean(np.abs(z) > 2.5))},
     "resultados": {}}

# ---------- Simulación ----------
rng = np.random.default_rng(SEMILLA)
for est in ("D_extremos", "D_medio"):
    M = modelo(est); R[f"modelo_{est}"] = M
    for forma in ("normal", "empirica"):
        principal = (est, forma) == PRINCIPAL; n_mc = N_MC if principal else N_MC_ROBUSTEZ
        for top in TOPS:
            k = int(round(top * N))
            for nombre, f in ESTRATEGIAS.items():
                if nombre == "frontera_homo" and not principal: continue
                for b in PRESUPUESTOS:
                    if nombre == "uniforme" and b != int(b): continue
                    if nombre != "uniforme" and b == 1.0: continue           # con una corrida todas coinciden con la uniforme
                    rec = []
                    for _ in range(n_mc):
                        mu = verdaderos(est, M, forma, rng)
                        rec.append(len(np.intersect1d(mejores(mu, k), f(mu, k, b, M, rng))) / k)
                    R["resultados"][f"{est}|{forma}|{top}|{nombre}|{b}"] = {"recall": float(np.mean(rec)), "ee": float(np.std(rec, ddof=1) / np.sqrt(n_mc))}
            linea = "  ".join(f"{n_}@{b:g}: {R['resultados'][f'{est}|{forma}|{top}|{n_}|{b}']['recall']:.2f}"
                              for n_ in ESTRATEGIAS for b in PRESUPUESTOS if f"{est}|{forma}|{top}|{n_}|{b}" in R["resultados"])
            print(f"{est} {forma} top{top:.0%}: {linea}", flush=True)


def rec(est, forma, top, nombre, b):
    clave = f"{est}|{forma}|{top}|{'uniforme' if b == 1.0 else nombre}|{b}"
    return R["resultados"][clave]["recall"] if clave in R["resultados"] else None


# presupuesto que necesita la frontera para igualar a 2, 3 y 4 réplicas uniformes
R["equivalencia"] = {}
for est in ("D_extremos", "D_medio"):
    for forma in ("normal", "empirica"):
        for top in TOPS:
            Bs = [b for b in PRESUPUESTOS if rec(est, forma, top, "frontera", b) is not None]
            fr = np.maximum.accumulate([rec(est, forma, top, "frontera", b) for b in Bs])    # monótona para interpolar
            for obj in (2.0, 3.0, 4.0):
                meta = rec(est, forma, top, "uniforme", obj)
                B_eq = float(np.interp(meta, fr, Bs)) if meta <= fr[-1] else None
                R["equivalencia"][f"{est}|{forma}|{top}|{obj:g}"] = {"recall_uniforme": meta, "presupuesto_frontera": B_eq,
                                                                     "ahorro": obj / B_eq if B_eq else None}
json.dump(R, open("resultados/asignacion.json", "w"), indent=1)

# ---------- Tabla, macros y figura (caso principal) ----------
est, forma = PRINCIPAL
B_TABLA = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
NOM = {"uniforme": "Uniform replicates", "mitades": "Successive halving", "frontera": "Boundary-focused"}
with open("paper/tabla_asignacion.tex", "w") as fh:
    fh.write("% Generado por asignacion.py. No editar a mano.\n\\begin{tabular}{ll" + "r" * len(B_TABLA) + "}\n\\toprule\n"
             " & & \\multicolumn{" + str(len(B_TABLA)) + "}{c}{Mean runs per candidate} \\\\\n\\cmidrule(lr){3-" + str(len(B_TABLA) + 2) + "}\n"
             "Target & Allocation & " + " & ".join(f"{b:g}" for b in B_TABLA) + " \\\\\n\\midrule\n")
    for j, top in enumerate(TOPS):
        if j: fh.write("\\addlinespace\n")
        for i, nombre in enumerate(NOM):
            celdas = [f"{100 * v:.0f}\\%" if (v := rec(est, forma, top, nombre, b)) is not None else "n/a" for b in B_TABLA]
            etiqueta = f"Top {100 * top:.0f}\\%" if i == 0 else ""
            fh.write(f"{etiqueta} & {NOM[nombre]} & " + " & ".join(celdas) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

f2 = lambda x: f"{x:.2f}"; pc = lambda x: f"{100 * x:.0f}"
G = R["gaussianidad"]
M = {"alN": f"{N:,}".replace(",", "{,}"), "alReps": str(N_MC), "alKurt": f2(G["curtosis_exceso"]), "alShapiroP": f2(G["shapiro_p"])}
for top, nt in ((0.01, "One"), (0.05, "Five"), (0.10, "Ten")):
    for b, nb in ((1.0, "One"), (2.0, "Two"), (4.0, "Four")):
        for nombre, ne in (("uniforme", "Unif"), ("mitades", "Halving"), ("frontera", "Bound")):
            v = rec(est, forma, top, nombre, b)
            if v is not None and not (b == 1.0 and nombre != "uniforme"): M[f"al{ne}Top{nt}B{nb}"] = pc(v)
    for obj, no in ((2.0, "Two"), (3.0, "Three"), (4.0, "Four")):
        q = R["equivalencia"][f"{est}|{forma}|{top}|{obj:g}"]
        M[f"alEqTop{nt}Unif{no}"] = f"{q['presupuesto_frontera']:.1f}" if q["presupuesto_frontera"] else "n/a"
        M[f"alSavingTop{nt}Unif{no}"] = f"{q['ahorro']:.1f}" if q["ahorro"] else "n/a"
M["alBoundHomoTopTenBFour"] = pc(rec(est, forma, 0.10, "frontera_homo", 4.0))
for e2, f2_, ne in (("D_extremos", "empirica", "EmpEnd"), ("D_medio", "normal", "NormAvg")):
    q = R["equivalencia"][f"{e2}|{f2_}|0.1|4"]; M[f"alSavingTopTenUnifFour{ne}"] = f"{q['ahorro']:.1f}" if q["ahorro"] else "n/a"
# halving (con rondas reiniciadas) frente a frontera: los dos casos citados en el texto y la información suplementaria
for e2, b2, top2, ne in (("D_extremos", 1.25, 0.01, "TopOneBOneQuarterEnd"), ("D_medio", 4.0, 0.01, "TopOneBFourAvg")):
    M[f"alHalving{ne}"] = pc(rec(e2, "normal", top2, "mitades", b2)); M[f"alBound{ne}"] = pc(rec(e2, "normal", top2, "frontera", b2))
with open("paper/numeros_asignacion.tex", "w") as fh:
    fh.write("% Generado por asignacion.py a partir de resultados/asignacion.json. No editar a mano.\n")
    for kk, v in M.items(): fh.write(f"\\newcommand{{\\{kk}}}{{{v}}}\n")

fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
for ax, top in zip(axs, TOPS):
    for nombre, color in (("uniforme", NARANJA), ("mitades", VERDE), ("frontera", AZUL)):
        xs = [b for b in PRESUPUESTOS if rec(est, forma, top, nombre, b) is not None]
        ax.plot(xs, [100 * rec(est, forma, top, nombre, b) for b in xs], color=color, marker="o", ms=3, label=NOM[nombre])
    ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 6]); ax.set_xticklabels(["1", "2", "4", "6"]); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_title(f"Top {100 * top:.0f}% of {N:,} candidates", fontsize=8.5); ax.set_xlabel("Mean 5 ns runs per candidate")
axs[0].set_ylabel("Recall of the true top set (%)")
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=8)
fig.tight_layout(); fig.savefig("paper/figures/fig8_allocation.pdf"); fig.savefig("paper/figures/fig8_allocation.png"); plt.close(fig)

print(f"\ngaussianidad de las réplicas: curtosis {G['curtosis_exceso']:.2f}, Shapiro p {G['shapiro_p']:.2f}")
for clave, q in R["equivalencia"].items():
    if q["presupuesto_frontera"]:
        print(f"{clave}: igualar uniforme (recall {q['recall_uniforme']:.2f}) con frontera b ≈ {q['presupuesto_frontera']:.2f} → ahorro ×{q['ahorro']:.2f}")
    else:
        print(f"{clave}: la frontera no llega con estos presupuestos")
