"""Análisis por segmentos de las simulaciones de 50 ns: ¿es el comienzo de una trayectoria más rápido que el resto?

Motivación: las corridas cortas de 5 ns (que parten de la misma estructura inicial que la larga del mismo sistema) dan una
difusividad aparente mayor que la que da, al mismo retardo, la curva de la trayectoria larga promediada en orígenes. Si el
comienzo de la trayectoria es más rápido que su media (equilibrado incompleto o envejecimiento de la matriz polimérica), todas
las corridas cortas del conjunto llevan ese sesgo.

Para cada trayectoria de 50 ns (data/iones/<id>.npz, posiciones − centro de masas, como en analizar_prefijos.py) se parte la
simulación en segmentos CONSECUTIVOS de 5 ns y, en cada uno y con sus datos solamente, se calcula:
  extremos   desplazamiento inicio→fin del segmento / (6·5 ns)          (el estimador de las corridas cortas de HTP-MD)
  mitad      MSD promediado en orígenes dentro del segmento a 2,5 ns / (6·2,5 ns)
  beta       exponente efectivo del MSD del segmento entre 1 y 3 ns (ventana 0,2–0,6, como en el texto)
Agregado (sobre sistemas, intervalos bootstrap por sistemas):
  - perfil: D del segmento k / media geométrica de los segmentos tardíos (25–50 ns), mediana por k;
  - exceso inicial: primer segmento frente a los tardíos (mediana, fracción de sistemas > 1, Wilcoxon);
  - deriva continuada: pendiente de ln D frente al tiempo en los segmentos 5–50 ns, por sistema;
  - comprobación: primer segmento frente a la media de TODOS los segmentos, que es lo que separa una corrida corta
    (o un prefijo) de la curva promediada en orígenes; se compara con el cociente observado en el texto.

Incremental: un JSON por trayectoria en resultados/segmentos/ (se puede agregar sin data/).
Uso: python segmentos.py → resultados/segmentos.json, paper/numeros_segmentos.tex, paper/figures/fig7_segments.pdf
"""
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEST = os.environ.get("HTPMD_DEST", "data/iones")
SEG_NS, N_BOOT, SEMILLA = 5.0, 2000, 20261016
TARDIOS_DESDE = 25.0                                             # ns: segmentos tomados como referencia "tardía"
AZUL, NARANJA, TINTA, TINTA2, REJ = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": REJ, "grid.linewidth": 0.6,
                     "lines.linewidth": 2.0, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})


def msd_por_ion(r):
    """MSD promediado en orígenes para cada ion (FFT); misma función que analizar_prefijos.py. r: (frames, iones, 3)."""
    x = np.transpose(r, (1, 0, 2)).astype(np.float64); n = x.shape[1]
    F = np.fft.rfft(x, n=2 * n, axis=1); s2 = np.fft.irfft(F * F.conj(), axis=1)[:, :n, :].sum(-1)
    d = np.square(x).sum(-1); c = np.cumsum(d, axis=1)
    m = np.arange(n)
    primero = c[:, n - 1 - m]
    segundo = c[:, -1][:, None] - np.concatenate([np.zeros((len(c), 1)), c[:, :-1]], axis=1)[:, m]
    return (primero + segundo - 2 * s2) / (n - m)


def analizar(f):
    d = np.load(f); dt = float(d["dt_ps"]) / 1000.0
    li = d["li"].astype(np.float64) - d["com"][:, None, :]
    total = (li.shape[0] - 1) * dt; paso = int(round(SEG_NS / dt))
    segs = []
    for k in range(int(total // SEG_NS + 1e-9)):
        r = li[k * paso: (k + 1) * paso + 1]
        t = np.arange(len(r)) * dt; msd = msd_por_ion(r).mean(0)
        w = (t >= 0.2 * SEG_NS) & (t <= 0.6 * SEG_NS) & (msd > 0)
        segs.append({"inicio_ns": k * SEG_NS,
                     "extremos": float(np.square(r[-1] - r[0]).sum(-1).mean() / (6 * SEG_NS)),
                     "mitad": float(np.interp(SEG_NS / 2, t, msd) / (3 * SEG_NS)),
                     "beta": float(np.polyfit(np.log(t[w]), np.log(msd[w]), 1)[0])})
    return {"ns": total, "n_li": int(li.shape[1]), "segmentos": segs}


os.makedirs("resultados/segmentos", exist_ok=True); os.makedirs("paper/figures", exist_ok=True)
for f in sorted(glob.glob(f"{DEST}/*.npz")):
    if "parcial" in f: continue
    cache = f"resultados/segmentos/{os.path.basename(f)[:-4]}.json"
    if not os.path.exists(cache):
        json.dump(analizar(f), open(cache, "w")); print("analizada", os.path.basename(f), flush=True)
todas = {}
for cache in sorted(glob.glob("resultados/segmentos/*.json")):
    r = json.load(open(cache))
    if r["ns"] >= 49: todas[os.path.basename(cache)[:-5]] = r
N = len(todas); print(N, "trayectorias de 50 ns con segmentos")
if N == 0: raise SystemExit("sin datos: ejecutar donde estén data/iones/*.npz (o HTPMD_DEST)")

ids = list(todas); n_seg = min(len(todas[i]["segmentos"]) for i in ids)
inicio = np.array([todas[ids[0]]["segmentos"][k]["inicio_ns"] for k in range(n_seg)]); tardio = inicio >= TARDIOS_DESDE
X = {e: np.log(np.array([[todas[i]["segmentos"][k][e] for k in range(n_seg)] for i in ids])) for e in ("extremos", "mitad")}
B = np.array([[todas[i]["segmentos"][k]["beta"] for k in range(n_seg)] for i in ids])
T = pd.read_csv("resultados/tabla_por_trayectoria.csv").set_index("id")
lit = np.array([bool(T.loc[int(i), "literatura"]) if int(i) in T.index else False for i in ids])
rng = np.random.default_rng(SEMILLA)
ic = lambda b: [float(np.nanquantile(b, .025)), float(np.nanquantile(b, .975))]


def resumen(x):
    """x: (sistemas, segmentos) en ln D. Cocientes respecto a los segmentos tardíos y a la media de todos."""
    rel = x - x[:, tardio].mean(1, keepdims=True)
    t = inicio[1:] + SEG_NS / 2                                            # deriva continuada sin el primer segmento
    pend = np.array([np.polyfit(t, fila[1:], 1)[0] * 10 for fila in x])    # ln D por cada 10 ns
    return {"perfil": np.exp(np.median(rel, 0)), "primero_vs_tardios": float(np.exp(np.median(rel[:, 0]))),
            "primero_vs_todos": float(np.exp(np.median(x[:, 0] - x.mean(1)))),
            "frac_primero_mayor": float((rel[:, 0] > 0).mean()), "pendiente_10ns": float(np.exp(np.median(pend))),
            "frac_pendiente_negativa": float((pend < 0).mean())}


R = {"n": N, "seg_ns": SEG_NS, "tardios_desde_ns": TARDIOS_DESDE, "inicio_ns": inicio.tolist()}
for e in ("extremos", "mitad"):
    x = X[e]; m = resumen(x); boot = []
    for _ in range(N_BOOT): boot.append(resumen(x[rng.integers(0, N, N)]))
    sal = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
    sal["perfil_ic95"] = [ic([b["perfil"][k] for b in boot]) for k in range(n_seg)]
    for k in ("primero_vs_tardios", "primero_vs_todos", "pendiente_10ns"):
        sal[f"{k}_ic95"] = ic([b[k] for b in boot])
    sal["wilcoxon_p"] = float(stats.wilcoxon(x[:, 0] - x[:, tardio].mean(1)).pvalue)
    for nombre, sel in (("cribado", ~lit), ("literatura", lit)):
        if sel.sum() >= 5: sal[f"primero_vs_tardios_{nombre}"] = float(np.exp(np.median(x[sel, 0] - x[sel][:, tardio].mean(1))))
    rel = np.exp(np.median(x - x[:, tardio].mean(1, keepdims=True), 0))
    dentro = np.where(rel < 1.10)[0]
    sal["primer_segmento_dentro_10pct_ns"] = float(inicio[dentro[0]]) if len(dentro) else None
    R[e] = sal
R["beta_mediana_por_segmento"] = np.median(B, 0).tolist()
json.dump(R, open("resultados/segmentos.json", "w"), indent=1)

f2 = lambda v: f"{v:.2f}"
M = {"segN": str(N)}
for e, nom in (("extremos", "End"), ("mitad", "Half")):
    s = R[e]
    M[f"seg{nom}First"] = f2(s["primero_vs_tardios"]); M[f"seg{nom}FirstLo"], M[f"seg{nom}FirstHi"] = (f2(v) for v in s["primero_vs_tardios_ic95"])
    M[f"seg{nom}FirstVsAll"] = f2(s["primero_vs_todos"]); M[f"seg{nom}FracFirstPct"] = f"{100 * s['frac_primero_mayor']:.0f}"
    M[f"seg{nom}Slope"] = f2(s["pendiente_10ns"]); M[f"seg{nom}FracNegSlopePct"] = f"{100 * s['frac_pendiente_negativa']:.0f}"
    M[f"seg{nom}WithinTen"] = f"{s['primer_segmento_dentro_10pct_ns']:.0f}" if s["primer_segmento_dentro_10pct_ns"] is not None else "n/a"
    mant, exp_ = f"{s['wilcoxon_p']:.1e}".split("e"); M[f"seg{nom}WilcoxonP"] = f"{mant}\\times10^{{{int(exp_)}}}"   # para modo matemático
    for nombre, nn in (("cribado", "Screen"), ("literatura", "Lit")):
        if f"primero_vs_tardios_{nombre}" in s: M[f"seg{nom}First{nn}"] = f2(s[f"primero_vs_tardios_{nombre}"])
M["segBetaFirst"] = f2(R["beta_mediana_por_segmento"][0]); M["segBetaLate"] = f2(np.median(np.array(R["beta_mediana_por_segmento"])[tardio]))
with open("paper/numeros_segmentos.tex", "w") as fh:
    fh.write("% Generado por segmentos.py a partir de resultados/segmentos.json. No editar a mano.\n")
    for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

fig, ax = plt.subplots(figsize=(4.2, 2.9))
for e, color, nombre in (("extremos", NARANJA, "End-point"), ("mitad", AZUL, "Origin-averaged, lag 2.5 ns")):
    s = R[e]; c = inicio + SEG_NS / 2; lo, hi = np.array(s["perfil_ic95"]).T
    ax.fill_between(c, lo, hi, color=color, alpha=0.18, lw=0); ax.plot(c, s["perfil"], color=color, marker="o", ms=3.5, label=nombre)
ax.axhline(1, color=TINTA2, lw=0.8, ls=(0, (3, 3))); ax.axvspan(TARDIOS_DESDE, inicio[-1] + SEG_NS, color=REJ, alpha=0.5, lw=0)
ax.set_xlabel("Segment centre (ns)"); ax.set_ylabel("$D_{app}$ of 5 ns segment / late segments")
ax.legend(frameon=False, fontsize=8)
fig.savefig("paper/figures/fig7_segments.pdf"); fig.savefig("paper/figures/fig7_segments.png"); plt.close(fig)

for e in ("extremos", "mitad"):
    s = R[e]
    print(f"\n{e}: perfil (D_k / tardíos) = {[round(v, 2) for v in s['perfil']]}")
    print(f"   primer segmento / tardíos ×{s['primero_vs_tardios']:.2f} {[round(v, 2) for v in s['primero_vs_tardios_ic95']]} "
          f"(mayor en {s['frac_primero_mayor']:.0%} de los sistemas, Wilcoxon p = {s['wilcoxon_p']:.1e})")
    print(f"   primer segmento / media de todos ×{s['primero_vs_todos']:.2f}  | deriva 5–50 ns: ×{s['pendiente_10ns']:.2f} por 10 ns "
          f"(negativa en {s['frac_pendiente_negativa']:.0%}) | dentro del 10 % desde {s['primer_segmento_dentro_10pct_ns']} ns")
print("\nβ mediana por segmento:", [round(v, 2) for v in R["beta_mediana_por_segmento"]])
