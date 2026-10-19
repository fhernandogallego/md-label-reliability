"""Análisis por prefijos de las simulaciones de 50 ns: ¿qué habría obtenido quien parase a T ns?

Para cada trayectoria (data/iones/<id>.npz) y cada duración T del prefijo se calculan cuatro estimadores de la
difusividad del Li a partir SOLO de los primeros T ns:
  extremos   desplazamiento primer→último fotograma / 6T            (el que usa HTP-MD)
  mitad      MSD promediado en orígenes a retardo T/2 / (6·T/2)
  ajuste     pendiente de un ajuste lineal del MSD en [0,2T ; 0,5T] / 6   (práctica habitual)
  potencial  ajuste MSD = A·τ^β en [0,1T ; 0,5T] extrapolado al retardo de referencia (predicción temprana)
Referencia: difusividad aparente a retardo 25 ns con la trayectoria completa de 50 ns.
Incertidumbre: bootstrap sobre iones. Reglas de parada: por régimen (β ≥ umbral) y por precisión (CV < umbral).

Incremental: guarda un JSON por trayectoria en resultados/prefijos/ y solo calcula las nuevas.
Uso: python analizar_prefijos.py   → resultados/prefijos.json y figuras en paper/figures/
"""
import glob, json, os
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEST = os.environ.get("HTPMD_DEST", "data/iones")
PREFIJOS = [1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50]          # ns
REJILLA_PARADA = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
TAU_REF, N_BOOT, SEMILLA = 25.0, 200, 20261009
ESTIMADORES = ["extremos", "mitad", "ajuste", "potencial"]
AZUL, NARANJA, VERDE, AMARILLO, TINTA, TINTA2, REJ = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = dict(zip(ESTIMADORES, [NARANJA, AMARILLO, AZUL, VERDE]))
NOMBRE = {"extremos": "End-point", "mitad": "Half-lag", "ajuste": "Linear fit", "potencial": "Power-law extrapolation"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": REJ, "grid.linewidth": 0.6,
                     "lines.linewidth": 2.0, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})


def msd_por_ion(r):
    """MSD promediado en orígenes de tiempo para cada ion (algoritmo FFT). r: (frames, iones, 3) → (iones, frames)."""
    x = np.transpose(r, (1, 0, 2)).astype(np.float64); n = x.shape[1]
    F = np.fft.rfft(x, n=2 * n, axis=1); s2 = np.fft.irfft(F * F.conj(), axis=1)[:, :n, :].sum(-1)
    d = np.square(x).sum(-1); c = np.cumsum(d, axis=1)
    m = np.arange(n)
    primero = c[:, n - 1 - m]                                     # suma de d[0 .. n-m-1]
    segundo = c[:, -1][:, None] - np.concatenate([np.zeros((len(c), 1)), c[:, :-1]], axis=1)[:, m]   # suma de d[m .. n-1]
    return (primero + segundo - 2 * s2) / (n - m)


def estimadores(curva, t, T):
    """curva: MSD medio (Å²) en retardos t (ns) del prefijo de duración T. Devuelve D en Å²/ns y el exponente local."""
    out = {"mitad": float(np.interp(T / 2, t, curva) / (3 * T))}
    k = (t >= 0.2 * T) & (t <= 0.5 * T)
    out["ajuste"] = float(np.polyfit(t[k], curva[k], 1)[0] / 6)
    k = (t >= 0.1 * T) & (t <= 0.5 * T) & (curva > 0)
    b, lnA = np.polyfit(np.log(t[k]), np.log(curva[k]), 1)
    out["potencial"] = float(np.exp(lnA) * TAU_REF ** (b - 1) / 6)
    k = (t >= 0.2 * T) & (t <= 0.6 * T) & (curva > 0)
    out["beta"] = float(np.polyfit(np.log(t[k]), np.log(curva[k]), 1)[0])
    return out


def analizar(f):
    d = np.load(f); dt = float(d["dt_ps"]) / 1000.0; rng = np.random.default_rng(SEMILLA)
    li = d["li"].astype(np.float64) - d["com"][:, None, :]                          # se resta la deriva del centro de masas, como HTP-MD
    n_iones = li.shape[1]; total = (li.shape[0] - 1) * dt
    idx_boot = rng.integers(0, n_iones, (N_BOOT, n_iones))
    completo = msd_por_ion(li).mean(0); t_all = np.arange(li.shape[0]) * dt
    res = {"ns": total, "n_li": n_iones, "ref": float(np.interp(TAU_REF, t_all, completo) / (6 * TAU_REF)), "prefijos": {},
           "deriva_com_A": float(np.linalg.norm(d["com"][-1] - d["com"][0]))}
    curva = f"data/curvas/{os.path.basename(f)[:-4]}.json"                           # validación frente a la curva distribuida por HTP-MD
    if os.path.exists(curva):
        c = json.load(open(curva))["li"]; tc = np.array(c["time"], float); tc = tc / tc[-1] * total
        k = tc <= 0.8 * total                                                       # el final de la curva tiene muy pocos orígenes
        res["validacion_max_desvio"] = float(np.abs(np.interp(tc[k], t_all, completo) / np.array(c["msd"], float)[k] - 1).max())
        sin = msd_por_ion(d["li"].astype(np.float64)).mean(0)
        res["inflado_sin_corregir_25ns"] = float(np.interp(TAU_REF, t_all, sin) / np.interp(TAU_REF, t_all, completo)) if total >= TAU_REF else None
    for T in sorted(set(PREFIJOS) | set(REJILLA_PARADA)):
        if T > total + 1e-9: continue
        nf = int(round(T / dt)) + 1; r = li[:nf]
        por_ion = msd_por_ion(r); t = np.arange(nf) * dt
        sel = np.unique(np.round(np.geomspace(1, nf - 1, 80)).astype(int))         # retardos en escala logarítmica
        ts, pi = t[sel], por_ion[:, sel]
        desp = np.square(r[-1] - r[0]).sum(-1)                                      # desplazamiento primer→último, por ion
        e = estimadores(pi.mean(0), ts, T); e["extremos"] = float(desp.mean() / (6 * T))
        boot = {k: [] for k in ESTIMADORES}
        for ib in idx_boot:
            eb = estimadores(pi[ib].mean(0), ts, T)
            for k in ("mitad", "ajuste", "potencial"): boot[k].append(eb[k])
            boot["extremos"].append(desp[ib].mean() / (6 * T))
        for k in ESTIMADORES:
            b = np.array(boot[k]); e[f"{k}_ic"] = [float(np.quantile(b, .025)), float(np.quantile(b, .975))]; e[f"{k}_cv"] = float(b.std() / abs(b.mean()))
        res["prefijos"][str(T)] = e
    return res


os.makedirs("resultados/prefijos", exist_ok=True); os.makedirs("paper/figures", exist_ok=True)
todas = {}
for f in sorted(glob.glob(f"{DEST}/*.npz")):
    if "parcial" in f: continue
    i = os.path.basename(f)[:-4]; cache = f"resultados/prefijos/{i}.json"
    if not os.path.exists(cache):
        r = analizar(f); json.dump(r, open(cache, "w")); print("analizada", i, flush=True)
    r = json.load(open(cache))
    if r["ns"] >= 49: todas[i] = r                                                  # solo las de 50 ns
for cache in sorted(glob.glob("resultados/prefijos/*.json")):                       # cachés sin trayectoria local (data/ fuera de git)
    i = os.path.basename(cache)[:-5]
    if i not in todas:
        r = json.load(open(cache))
        if r["ns"] >= 49: todas[i] = r
N = len(todas); print(N, "trayectorias de 50 ns analizadas")
val = np.array([todas[i].get("validacion_max_desvio", np.nan) for i in todas]); infl = np.array([todas[i].get("inflado_sin_corregir_25ns") or np.nan for i in todas])
print(f"validación frente a las curvas de HTP-MD: desvío máximo mediano {np.nanmedian(val):.2e}, peor caso {np.nanmax(val):.2e}")
print(f"si NO se resta la deriva del centro de masas, el MSD a 25 ns se infla ×{np.nanmedian(infl):.2f} de mediana (máx ×{np.nanmax(infl):.1f})")
ref = np.array([todas[i]["ref"] for i in todas])

# ---------- Estimadores frente a la duración del prefijo ----------
R = {"n": N, "tau_ref": TAU_REF, "prefijos_ns": PREFIJOS, "estimadores": {},
     "validacion": {"desvio_max_mediano": float(np.nanmedian(val)), "desvio_max_peor": float(np.nanmax(val)),
                    "inflado_sin_deriva_mediana": float(np.nanmedian(infl)), "inflado_sin_deriva_max": float(np.nanmax(infl)),
                    "frac_inflado_mayor_10pct": float(np.nanmean(infl > 1.1))}}
for k in ESTIMADORES:
    filas = []
    for T in PREFIJOS:
        e = np.array([todas[i]["prefijos"][str(T)][k] for i in todas]); ic = np.array([todas[i]["prefijos"][str(T)][f"{k}_ic"] for i in todas])
        ok = e > 0; lr = np.log(e[ok] / ref[ok])
        filas.append({"T": T, "razon_mediana": float(np.exp(np.median(lr))), "p10": float(np.exp(np.quantile(lr, .1))), "p90": float(np.exp(np.quantile(lr, .9))),
                      "factor_error": float(np.exp(np.median(np.abs(lr)))),            # error multiplicativo típico
                      "spearman": float(stats.spearmanr(e, ref)[0]) if N > 3 else None,
                      "cobertura_ic95": float(((ic[:, 0] <= ref) & (ref <= ic[:, 1])).mean()),
                      "frac_no_positivo": float((~ok).mean())})
    R["estimadores"][k] = filas
beta = {T: np.array([todas[i]["prefijos"][str(T)]["beta"] for i in todas]) for T in PREFIJOS}
R["beta_por_prefijo"] = [{"T": T, "mediana": float(np.median(beta[T])), "frac_difusivo_0.9": float((beta[T] >= 0.9).mean())} for T in PREFIJOS]

# ---------- Reglas de parada ----------
def regla(nombre, condicion, estimador):
    """Una corrida que no cumple la regla antes de 50 ns se simula entera y se evalúa con el estimador a 50 ns: el error y el
    tiempo medio se calculan sobre TODAS las corridas, no solo sobre las que paran (eso favorecería a las reglas que casi nunca paran)."""
    paradas, err_todas, err_paran = [], [], []
    for i in todas:
        T_stop = next((T for T in REJILLA_PARADA if T < 50 and condicion(todas[i]["prefijos"][str(T)])), None)
        paradas.append(T_stop)
        e = abs(np.log(todas[i]["prefijos"][str(T_stop or 50)][estimador] / todas[i]["ref"]))
        err_todas.append(e)
        if T_stop is not None: err_paran.append(e)
    par = [p for p in paradas if p is not None]
    return {"regla": nombre, "estimador": estimador, "frac_para": len(par) / N, "T_medio_si_para": float(np.mean(par)) if par else None,
            "T_medio_total": float(np.mean([p if p is not None else 50 for p in paradas])),
            "factor_error": float(np.exp(np.median(err_todas))),
            "factor_error_si_para": float(np.exp(np.median(err_paran))) if err_paran else None}
R["reglas"] = [regla("regimen_beta_0.9", lambda e: e["beta"] >= 0.9, "ajuste"),
               regla("regimen_beta_0.8", lambda e: e["beta"] >= 0.8, "ajuste"),
               regla("precision_cv_10", lambda e: e["ajuste_cv"] < 0.10, "ajuste"),
               regla("precision_cv_20", lambda e: e["ajuste_cv"] < 0.20, "ajuste"),
               regla("precision_cv_10_potencial", lambda e: e["potencial_cv"] < 0.10, "potencial")]
for k, Ts in (("extremos", (5,)), ("ajuste", (5, 10, 20)), ("potencial", (5, 10, 20))):   # extremos a 5 ns = práctica de HTP-MD
    for T in Ts:
        fe = next(x["factor_error"] for x in R["estimadores"][k] if x["T"] == T)
        R["reglas"].append({"regla": f"fijo_{T}ns", "estimador": k, "frac_para": 1.0, "T_medio_si_para": T, "T_medio_total": T,
                            "factor_error": fe, "factor_error_si_para": fe})
json.dump(R, open("resultados/prefijos.json", "w"), indent=1)

# ---------- Tablas para el manuscrito (generadas, no editar a mano) ----------
fila_T = lambda k, T: next(x for x in R["estimadores"][k] if x["T"] == T)
with open("paper/tabla_prefijos.tex", "w") as fh:
    fh.write("% Generado por analizar_prefijos.py. No editar a mano.\n\\begin{tabular}{r" + "rr" * 4 + "}\n\\toprule\n")
    fh.write(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{NOMBRE[k]}}}" for k in ESTIMADORES) + " \\\\\n")
    fh.write("".join(f"\\cmidrule(lr){{{2 + 2 * j}-{3 + 2 * j}}}" for j in range(4)) + "\n")
    fh.write("$T$ (ns) & " + " & ".join("Ratio & Error" for _ in ESTIMADORES) + " \\\\\n\\midrule\n")
    for T in (1, 2, 5, 10, 20, 30, 50):
        fh.write(f"{T} & " + " & ".join(f"{fila_T(k, T)['razon_mediana']:.2f} & $\\times${fila_T(k, T)['factor_error']:.2f}" for k in ESTIMADORES) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
ETIQ = {"regimen_beta_0.9": "Regime: $\\beta\\ge0.9$", "regimen_beta_0.8": "Regime: $\\beta\\ge0.8$",
        "precision_cv_10": "Precision: CV $<10\\%$", "precision_cv_20": "Precision: CV $<20\\%$",
        "precision_cv_10_potencial": "Precision: CV $<10\\%$",
        "fijo_5ns": "Fixed \\SI{5}{\\nano\\second}", "fijo_10ns": "Fixed \\SI{10}{\\nano\\second}", "fijo_20ns": "Fixed \\SI{20}{\\nano\\second}"}
EST_CORTO = {"extremos": "End-point", "mitad": "Half-lag", "ajuste": "Linear fit", "potencial": "Power law"}
with open("paper/tabla_reglas.tex", "w") as fh:
    fh.write("% Generado por analizar_prefijos.py. No editar a mano.\n\\begin{tabular}{llrrrr}\n\\toprule\n"
             "Rule & Estimator & Stop before \\SI{50}{\\nano\\second} & Mean time (ns) & Error, all runs & Error, runs that stop \\\\\n\\midrule\n")
    for j, x in enumerate(R["reglas"]):
        if j and x["regla"].startswith("fijo") and not R["reglas"][j - 1]["regla"].startswith("fijo"): fh.write("\\midrule\n")
        fijo = x["regla"].startswith("fijo")
        fp = f"$\\times${x['factor_error_si_para']:.2f}" if x["factor_error_si_para"] and not fijo else "n/a"
        fr = "n/a" if fijo else f"{100 * x['frac_para']:.0f}\\%"
        fh.write(f"{ETIQ[x['regla']]} & {EST_CORTO[x['estimador']]} & {fr} & {x['T_medio_total']:.1f} & $\\times${x['factor_error']:.2f} & {fp} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ---------- Figura: sesgo y ranking por estimador ----------
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
for k in ESTIMADORES:
    f = R["estimadores"][k]; T = [x["T"] for x in f]
    axs[0].plot(T, [x["razon_mediana"] for x in f], color=COLOR[k], marker="o", ms=3.5, label=NOMBRE[k])
    if N > 3: axs[1].plot(T, [x["spearman"] for x in f], color=COLOR[k], marker="o", ms=3.5)
axs[0].axhline(1, color=TINTA2, lw=0.8, ls=(0, (3, 3))); axs[0].set_yscale("log")
axs[0].set_ylabel("Estimate / reference (median)"); axs[1].set_ylabel("Spearman ρ with reference ranking")
for ax in axs:
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50"]); ax.set_xlabel("Simulated time T (ns)")
axs[0].set_yticks([0.5, 1, 2, 5, 10]); axs[0].set_yticklabels(["0.5", "1", "2", "5", "10"])
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=8, handlelength=1.4)
fig.tight_layout(); fig.savefig("paper/figures/fig5_prefix_estimators.pdf"); fig.savefig("paper/figures/fig5_prefix_estimators.png"); plt.close(fig)

print(f"\nReferencia: D_app a {TAU_REF:g} ns con la trayectoria completa. N = {N}")
for k in ESTIMADORES:
    print(f"\n{k}:  T(ns)  razón mediana [p10–p90]  error típico ×  Spearman  cobertura IC95")
    for x in R["estimadores"][k]:
        sp = f"{x['spearman']:.2f}" if x["spearman"] is not None else "  - "
        print(f"   {x['T']:>5}   {x['razon_mediana']:.2f} [{x['p10']:.2f}–{x['p90']:.2f}]      {x['factor_error']:.2f}        {sp}     {x['cobertura_ic95']:.0%}")
print("\nExponente por prefijo:", [(x["T"], round(x["mediana"], 2), f"{x['frac_difusivo_0.9']:.0%}") for x in R["beta_por_prefijo"]])
print("\nReglas de parada:")
for x in R["reglas"]:
    tm = f"{x['T_medio_si_para']:.1f}" if x["T_medio_si_para"] else "-"; fp = f"{x['factor_error_si_para']:.2f}" if x["factor_error_si_para"] else "-"
    print(f"   {x['regla']:28s} {x['estimador']:9s} para {x['frac_para']:.0%} | T medio si para {tm} ns | T medio total {x['T_medio_total']:.1f} ns"
          f" | error típico ×{x['factor_error']:.2f} (si para ×{fp})")
