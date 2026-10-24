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
Comprobaciones (al final): error Monte Carlo, fiabilidad mal estimada al planificar y re-cribado en dos etapas.
Uso: python asignacion.py → resultados/asignacion.json, resultados/asignacion_extra.json, paper/tabla_asignacion.tex,
     paper/tabla_asignacion_extra.tex, paper/numeros_asignacion.tex, paper/figures/fig8_allocation.pdf
     python asignacion.py extra    solo las comprobaciones (lee la simulación principal del JSON; ~15 min)
     python asignacion.py tablas   solo tablas, macros y figura a partir de los dos JSON
"""
import json, sys
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


# M: modelo de ruido real (genera las corridas); Mp: el que usa quien planifica y selecciona (por defecto el mismo; distinto
# solo en la sensibilidad a una fiabilidad mal estimada)
def uniforme(mu, k, b, M, rng, Mp=None):
    n = np.full(N, int(b)); suma = np.zeros(N)
    for _ in range(int(b)): suma += correr(mu, np.arange(N), M, rng)
    return mejores(posterior(suma, n, Mp or M)[0], k)


def mitades(mu, k, b, M, rng, Mp=None):
    Mp = Mp or M
    n = np.ones(N); suma = correr(mu, np.arange(N), M, rng); resto = int(round((b - 1) * N)); g = 4 * k
    while resto > 0:
        sel = mejores(posterior(suma, n, Mp)[0], min(g, resto, N - 1))
        suma[sel] += correr(mu, sel, M, rng); n[sel] += 1; resto -= len(sel)
        g = g // 2 if g // 2 >= k else 4 * k                                 # al llegar a k, nueva ronda desde 4k (no repetir siempre los mismos k)
    return mejores(posterior(suma, n, Mp)[0], k)


def frontera(mu, k, b, M, rng, homo=False, Mp=None):
    Mp = Mp or M
    n = np.ones(N); suma = correr(mu, np.arange(N), M, rng); resto = int(round((b - 1) * N))
    while resto > 0:
        m, s = posterior(suma, n, Mp, homo); c = np.partition(m, N - k)[N - k]
        p = stats.norm.cdf((m - c) / s); lote = min(int(TANDA * N), resto)
        sel = np.argpartition(-(p * (1 - p)), lote)[:lote]
        suma[sel] += correr(mu, sel, M, rng); n[sel] += 1; resto -= lote
    return mejores(posterior(suma, n, Mp, homo)[0], k)


def dos_etapas(mu, k, b, M, rng, lista):
    """Re-cribado clásico: una corrida a todos y el resto del presupuesto repartido por igual entre los `lista` mejores según
    la primera corrida (lo que no divide exacto, una corrida más a los primeros de la lista); selección por la media posterior."""
    n = np.ones(N); suma = correr(mu, np.arange(N), M, rng); resto = int(round((b - 1) * N))
    corta = mejores(suma, lista); corta = corta[np.argsort(-suma[corta])]
    for j in range(resto // lista): suma[corta] += correr(mu, corta, M, rng); n[corta] += 1
    extra = corta[:resto % lista]; suma[extra] += correr(mu, extra, M, rng); n[extra] += 1
    return mejores(posterior(suma, n, M)[0], k)


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
# "python asignacion.py extra" lee la simulación principal de resultados/asignacion.json y solo calcula las comprobaciones
# del final (error Monte Carlo, fiabilidad mal estimada, dos etapas): la simulación adaptativa difiere entre máquinas en
# algún punto de recall por la coma flotante, y así no se mueven las cifras ya publicadas.
SOLO_VERDAD = len(sys.argv) > 1 and sys.argv[1] == "verdad"        # "verdad": solo la comprobación 4 (fiabilidad verdadera)
SOLO_TABLAS = SOLO_VERDAD or (len(sys.argv) > 1 and sys.argv[1] == "tablas")   # "tablas": sin simular nada, solo tabla, macros y figura
SOLO_EXTRA = SOLO_TABLAS or (len(sys.argv) > 1 and sys.argv[1] == "extra")
rng = np.random.default_rng(SEMILLA)
for est in (() if SOLO_EXTRA else ("D_extremos", "D_medio")):
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


def igualar(recalls, Bs, meta):
    """Presupuesto (interpolado) con el que una estrategia alcanza el recall `meta`; None si no llega."""
    fr = np.maximum.accumulate(recalls)                                     # monótona para interpolar
    return float(np.interp(meta, fr, Bs)) if meta <= fr[-1] else None


if SOLO_EXTRA: R = json.load(open("resultados/asignacion.json"))
# presupuesto que necesita la frontera para igualar a 2, 3 y 4 réplicas uniformes
R["equivalencia"] = {}
for est in ("D_extremos", "D_medio"):
    for forma in ("normal", "empirica"):
        for top in TOPS:
            Bs = [b for b in PRESUPUESTOS if rec(est, forma, top, "frontera", b) is not None]
            for obj in (2.0, 3.0, 4.0):
                meta = rec(est, forma, top, "uniforme", obj)
                B_eq = igualar([rec(est, forma, top, "frontera", b) for b in Bs], Bs, meta)
                R["equivalencia"][f"{est}|{forma}|{top}|{obj:g}"] = {"recall_uniforme": meta, "presupuesto_frontera": B_eq,
                                                                     "ahorro": obj / B_eq if B_eq else None}
if not SOLO_EXTRA: json.dump(R, open("resultados/asignacion.json", "w"), indent=1)

# ---------- Comprobaciones (caso principal) ----------
#  1) error Monte Carlo de cada recall (ee = sd/√n_mc, ya guardado con cada resultado);
#  2) fiabilidad mal estimada: las corridas siguen el modelo medido (r = 0,47), pero quien planifica y selecciona usa los
#     extremos del IC 95 % de r. Se mantiene la dispersión total observada: σ_run² = (1−r)·σ_tot², σ_sis² = r·σ_tot², y la
#     pendiente con la movilidad se escala con σ_run. La uniforme no cambia (con el mismo n, el orden es el de las medias);
#  3) re-cribado en dos etapas, la referencia sencilla: una corrida a todos y el resto a una lista corta fija de los mejores
#     según esa corrida; lista de 2k, 4k u 8k, y se da la mejor de las tres para cada presupuesto (a favor de la referencia).
est0, forma0 = PRINCIPAL; M0 = modelo(est0); B_EXTRA = [b for b in PRESUPUESTOS if b > 1]
ee = [v["ee"] for c, v in R["resultados"].items() if c.startswith(f"{est0}|{forma0}|") and "homo" not in c]
X = {"n_mc": N_MC, "ee_max": float(max(ee)), "ee_mediana": float(np.median(ee)), "mal_estimada": {}, "dos_etapas": {}}
s_tot = M0["sd_run"] / np.sqrt(1 - M0["fiabilidad"])


def planificado(r):
    sr = s_tot * np.sqrt(1 - r); f = sr / M0["sd_run"]
    return {**M0, "fiabilidad": r, "sd_run": sr, "s_sis": s_tot * np.sqrt(r), "a": M0["a"] * f, "b": M0["b"] * f}


rng_x = np.random.default_rng(SEMILLA + 1)
for top in (() if SOLO_TABLAS else TOPS):
    k = int(round(top * N)); meta = rec(est0, forma0, top, "uniforme", 4.0)
    for r_p in U[est0]["fiabilidad_ic95"]:
        Mp = planificado(r_p); e = {"r_planificada": r_p}
        for nombre, f in (("frontera", frontera), ("mitades", mitades)):
            vals = []
            for b in B_EXTRA:
                rc = [len(np.intersect1d(mejores(mu, k), f(mu, k, b, M0, rng_x, Mp=Mp))) / k
                      for mu in (verdaderos(est0, M0, forma0, rng_x) for _ in range(N_MC))]
                e[f"{nombre}|{b}"] = {"recall": float(np.mean(rc)), "ee": float(np.std(rc, ddof=1) / np.sqrt(N_MC))}; vals.append(np.mean(rc))
            B_eq = igualar(vals, B_EXTRA, meta); e[f"{nombre}_ahorro_frente_a_4"] = 4.0 / B_eq if B_eq else None
        X["mal_estimada"][f"{top}|{r_p:.4f}"] = e
        print(f"top {top:.0%}, planificando con r = {r_p:.2f}: frontera b=2 {e['frontera|2.0']['recall']:.2f}, b=4 {e['frontera|4.0']['recall']:.2f}, "
              f"ahorro ×{e['frontera_ahorro_frente_a_4'] or float('nan'):.2f}", flush=True)
    e = {}
    for mult in (2, 4, 8):
        lista = min(mult * k, N); vals = []
        for b in B_EXTRA:
            rc = [len(np.intersect1d(mejores(mu, k), dos_etapas(mu, k, b, M0, rng_x, lista))) / k
                  for mu in (verdaderos(est0, M0, forma0, rng_x) for _ in range(N_MC))]
            e[f"{mult}k|{b}"] = {"recall": float(np.mean(rc)), "ee": float(np.std(rc, ddof=1) / np.sqrt(N_MC))}; vals.append(np.mean(rc))
    mejor = {b: max(e[f"{m}k|{b}"]["recall"] for m in (2, 4, 8)) for b in B_EXTRA}
    e["mejor_lista"] = {str(b): float(v) for b, v in mejor.items()}
    B_eq = igualar([mejor[b] for b in B_EXTRA], B_EXTRA, meta); e["ahorro_frente_a_4"] = 4.0 / B_eq if B_eq else None
    X["dos_etapas"][str(top)] = e
    print(f"top {top:.0%}, dos etapas (mejor lista): b=2 {mejor[2.0]:.2f}, b=4 {mejor[4.0]:.2f}, ahorro ×{e['ahorro_frente_a_4'] or float('nan'):.2f}", flush=True)
if SOLO_TABLAS: X = json.load(open("resultados/asignacion_extra.json"))
else: json.dump(X, open("resultados/asignacion_extra.json", "w"), indent=1)

#  4) incertidumbre de la fiabilidad en la propia verdad: las corridas siguen r en los extremos de su IC 95 % (con la misma
#     dispersión total, como en 2) y quien planifica conoce ese r; ahorro de la frontera frente a 4 réplicas uniformes
#     simuladas bajo la misma verdad. Junto con 2), acota el ahorro cuando r no se conoce con precisión.
if SOLO_VERDAD or not SOLO_TABLAS:
    V = {}; rng_v = np.random.default_rng(SEMILLA + 2)
    for r_v in U[est0]["fiabilidad_ic95"]:
        Mv = planificado(r_v)
        for top in TOPS:
            k = int(round(top * N))
            uni = float(np.mean([len(np.intersect1d(mejores(mu, k), uniforme(mu, k, 4.0, Mv, rng_v))) / k
                                 for mu in (verdaderos(est0, Mv, forma0, rng_v) for _ in range(N_MC))]))
            vals = [float(np.mean([len(np.intersect1d(mejores(mu, k), frontera(mu, k, b, Mv, rng_v))) / k
                                   for mu in (verdaderos(est0, Mv, forma0, rng_v) for _ in range(N_MC))])) for b in B_EXTRA]
            B_eq = igualar(vals, B_EXTRA, uni)
            V[f"{top}|{r_v:.4f}"] = {"r_verdadera": r_v, "uniforme|4.0": uni, **{f"frontera|{b}": v for b, v in zip(B_EXTRA, vals)},
                                     "presupuesto_frontera": B_eq, "ahorro_frente_a_4": 4.0 / B_eq if B_eq else None}
            print(f"top {top:.0%}, verdad r = {r_v:.2f}: uniforme×4 {uni:.2f}, frontera b=2 {vals[B_EXTRA.index(2.0)]:.2f}, "
                  f"ahorro ×{V[f'{top}|{r_v:.4f}']['ahorro_frente_a_4'] or float('nan'):.2f}", flush=True)
    json.dump(V, open("resultados/asignacion_verdad.json", "w"), indent=1)
else:
    import os
    V = json.load(open("resultados/asignacion_verdad.json")) if os.path.exists("resultados/asignacion_verdad.json") else {}

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
# comprobaciones: error Monte Carlo, fiabilidad mal estimada y dos etapas
sx = lambda v: f"{v:.1f}" if v else "n/a"
M["alMCErrMax"] = f"{100 * X['ee_max']:.1f}"; M["alMCErrMed"] = f"{100 * X['ee_mediana']:.1f}"
r_lo, r_hi = U[est0]["fiabilidad_ic95"]; M["alRelPlanLo"], M["alRelPlanHi"] = f2(r_lo), f2(r_hi); M["alRelTrue"] = f2(M0["fiabilidad"])
for top, nt in ((0.01, "One"), (0.05, "Five"), (0.10, "Ten")):
    for r_p, nr in ((r_lo, "Lo"), (r_hi, "Hi")):
        e = X["mal_estimada"][f"{top}|{r_p:.4f}"]
        M[f"alMis{nr}Top{nt}BTwo"] = pc(e["frontera|2.0"]["recall"]); M[f"alMis{nr}Top{nt}BFour"] = pc(e["frontera|4.0"]["recall"])
        M[f"alMis{nr}SavingTop{nt}"] = sx(e["frontera_ahorro_frente_a_4"])
    e = X["dos_etapas"][str(top)]
    M[f"alTwoStageTop{nt}BTwo"] = pc(e["mejor_lista"]["2.0"]); M[f"alTwoStageTop{nt}BFour"] = pc(e["mejor_lista"]["4.0"])
    M[f"alTwoStageSavingTop{nt}"] = sx(e["ahorro_frente_a_4"])
    for mult, nm in ((4, "Four"), (8, "Eight")):                          # lista fijada de antemano (no elegida a posteriori)
        B_eq = igualar([e[f"{mult}k|{b}"]["recall"] for b in B_EXTRA], B_EXTRA, rec(est0, forma0, top, "uniforme", 4.0))
        M[f"alTwoStage{nm}kSavingTop{nt}"] = sx(4.0 / B_eq if B_eq else None)
    for r_v, nr in ((r_lo, "Lo"), (r_hi, "Hi")):
        if f"{top}|{r_v:.4f}" in V: M[f"alTrue{nr}SavingTop{nt}"] = sx(V[f"{top}|{r_v:.4f}"]["ahorro_frente_a_4"])
with open("paper/tabla_asignacion_extra.tex", "w") as fh:
    fh.write("% Generado por asignacion.py. No editar a mano.\n\\begin{tabular}{lrrrr}\n\\toprule\n"
             f" & \\multicolumn{{3}}{{c}}{{Boundary-focused, planned with $r=$}} & Two-stage \\\\\n\\cmidrule(lr){{2-4}}\n"
             f"Target & {M['alRelTrue']} (true) & {M['alRelPlanLo']} & {M['alRelPlanHi']} & re-screening \\\\\n\\midrule\n")
    for top, nt in ((0.01, "One"), (0.05, "Five"), (0.10, "Ten")):
        celdas = [f"{M[f'alBoundTop{nt}BTwo']}\\% / {M[f'alBoundTop{nt}BFour']}\\% ($\\times${M[f'alSavingTop{nt}UnifFour']})"]
        celdas += [f"{M[f'alMis{nr}Top{nt}BTwo']}\\% / {M[f'alMis{nr}Top{nt}BFour']}\\% ($\\times${M[f'alMis{nr}SavingTop{nt}']})" for nr in ("Lo", "Hi")]
        celdas.append(f"{M[f'alTwoStageTop{nt}BTwo']}\\% / {M[f'alTwoStageTop{nt}BFour']}\\% ($\\times${M[f'alTwoStageSavingTop{nt}']})")
        fh.write(f"Top {100 * top:.0f}\\% & " + " & ".join(celdas) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
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
