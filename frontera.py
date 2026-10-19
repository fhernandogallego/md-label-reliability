"""Frontera coste–error de las reglas de parada y elección de umbrales evaluada en sistemas reservados.

Usa las cachés de analizar_prefijos.py (resultados/prefijos/<id>.json, solo las simulaciones de 50 ns). Para cada regla y
cada valor de su parámetro: tiempo medio simulado (las corridas que no paran antes de 50 ns cuentan 50 ns y se evalúan allí)
y error típico exp(mediana |ln(D_est / D_ref)|) frente a D_app(25 ns) de la simulación completa, igual que en el texto.

Reglas:
  fija-<estimador>        duración fija T (extremos, ajuste lineal, ley potencial)
  regimen                 parar cuando el exponente del prefijo alcanza β ≥ b; estimador: ajuste lineal
  precision-<estimador>   parar cuando el CV bootstrap del estimador baja de c (ajuste lineal o ley potencial)

Elección en reservados: se reparten los sistemas al azar en dos mitades (N_PART veces). En la mitad de ajuste se elige,
para cada familia, la opción más barata cuyo error no supera el objetivo ε; se mide su tiempo y su error en la otra mitad.

Uso: python frontera.py → resultados/frontera.json, paper/figures/fig6_cost_error.pdf, paper/tabla_frontera.tex, paper/numeros_frontera.tex
"""
import glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def err_log(est, ref):
    """|ln(est/ref)|; una estimación no positiva (p. ej. pendiente negativa a 1 ns) cuenta como fallo total."""
    return abs(np.log(est / ref)) if est > 0 else np.inf

N_PART, SEMILLA = 500, 20261016
OBJETIVOS = [1.5, 1.3, 1.2]
AZUL, NARANJA, VERDE, AMARILLO, MAGENTA, TINTA, TINTA2, REJ = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": REJ, "grid.linewidth": 0.6,
                     "lines.linewidth": 2.0, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})

runs = [json.load(open(f)) for f in sorted(glob.glob("resultados/prefijos/*.json"))]
runs = [r for r in runs if r["ns"] >= 49]
REJILLA = sorted({float(T) for r in runs for T in r["prefijos"]})
REJILLA = [T for T in REJILLA if all(str(T if T != int(T) else int(T)) in r["prefijos"] for r in runs)]
clave = lambda T: str(T if T != int(T) else int(T))
ref = np.array([r["ref"] for r in runs]); n = len(runs)


def evaluar(estimador, condicion):
    """Para cada corrida: (tiempo simulado, error log absoluto) con la regla dada."""
    t, e = np.empty(n), np.empty(n)
    for j, r in enumerate(runs):
        T = next((T for T in REJILLA if T < 50 and condicion(r["prefijos"][clave(T)])), 50.0)
        t[j] = T; e[j] = err_log(r["prefijos"][clave(T)][estimador], ref[j])
    return t, e


opciones = []                                                   # (familia, parámetro, tiempos, errores)
for est in ("extremos", "ajuste", "potencial"):
    for T in REJILLA:
        errores = np.array([err_log(r["prefijos"][clave(T)][est], ref[j]) for j, r in enumerate(runs)])
        opciones.append((f"fija-{est}", T, np.full(n, T), errores))
for b in np.round(np.arange(0.40, 0.951, 0.05), 2):
    opciones.append(("regimen", float(b), *evaluar("ajuste", lambda p, b=b: p["beta"] >= b)))
for est in ("ajuste", "potencial"):
    for c in np.round(np.arange(0.05, 0.501, 0.025), 3):
        opciones.append((f"precision-{est}", float(c), *evaluar(est, lambda p, c=c, est=est: p[f"{est}_cv"] < c)))

FAMILIAS = ["fija-extremos", "fija-ajuste", "fija-potencial", "regimen", "precision-ajuste", "precision-potencial"]
T_ = np.array([o[2] for o in opciones]); E_ = np.array([o[3] for o in opciones])
fam = np.array([o[0] for o in opciones]); par = np.array([o[1] for o in opciones])
todo = {"n": n, "rejilla_ns": REJILLA, "opciones": [{"familia": o[0], "parametro": o[1], "tiempo_medio": float(o[2].mean()),
                                                     "factor_error": float(np.exp(np.median(o[3])))} for o in opciones]}

# --- elección de umbral en una mitad, evaluación en la otra ---
rng = np.random.default_rng(SEMILLA); reserva = {}
for eps in OBJETIVOS:
    for f in FAMILIAS:
        idx = np.where(fam == f)[0]; filas = []
        for _ in range(N_PART):
            p = rng.permutation(n); a, b = p[: n // 2], p[n // 2:]
            t_a = T_[idx][:, a].mean(1); e_a = np.exp(np.median(E_[idx][:, a], axis=1))
            ok = np.where(e_a <= eps)[0]
            if len(ok) == 0: filas.append((np.nan, np.nan, 0.0, np.nan)); continue
            k = ok[np.argmin(t_a[ok])]
            t_b = T_[idx[k]][b].mean(); e_b = float(np.exp(np.median(E_[idx[k]][b])))
            filas.append((t_b, e_b, float(e_b <= eps), par[idx[k]]))
        x = np.array(filas, float)
        reserva[f"{f}|{eps}"] = {"frac_sin_opcion": float(np.isnan(x[:, 0]).mean()),
                                 "tiempo_test_mediana": float(np.nanmedian(x[:, 0])) if not np.isnan(x[:, 0]).all() else None,
                                 "error_test_mediana": float(np.nanmedian(x[:, 1])) if not np.isnan(x[:, 1]).all() else None,
                                 "frac_cumple_test": float(np.nanmean(x[:, 2])),
                                 "parametro_mediana": float(np.nanmedian(x[:, 3])) if not np.isnan(x[:, 3]).all() else None}
todo["reserva"] = reserva

# --- lo mínimo para alcanzar cada objetivo con todas las corridas (frontera) ---
todo["minimo"] = {}
for eps in OBJETIVOS:
    for f in FAMILIAS:
        idx = [i for i in np.where(fam == f)[0] if np.exp(np.median(E_[i])) <= eps]
        todo["minimo"][f"{f}|{eps}"] = float(min(T_[i].mean() for i in idx)) if idx else None
json.dump(todo, open("resultados/frontera.json", "w"), indent=1, default=float)

# --- figura ---
fig, ax = plt.subplots(figsize=(5.2, 3.4))
estilo = {"fija-extremos": (NARANJA, "-", "Fixed length, end-point"), "fija-ajuste": (AZUL, "-", "Fixed length, linear fit"),
          "fija-potencial": (VERDE, "-", "Fixed length, power law"), "regimen": (AMARILLO, (0, (4, 2)), "Regime rule ($\\beta\\geq b$), linear fit"),
          "precision-ajuste": (MAGENTA, (0, (1, 1.5)), "Precision rule (CV $<c$), linear fit"),
          "precision-potencial": (TINTA2, (0, (4, 2)), "Precision rule (CV $<c$), power law")}
for f in FAMILIAS:
    idx = np.where(fam == f)[0]; x = T_[idx].mean(1); y = np.exp(np.median(E_[idx], axis=1))
    o = np.argsort(x); col, ls, lab = estilo[f]
    ax.plot(x[o], y[o], color=col, ls=ls, lw=1.8 if f.startswith("fija") else 1.4, marker="o" if f.startswith("fija") else None, ms=3, label=lab)
for eps in (1.3,):
    ax.axhline(eps, color=TINTA2, lw=0.8, ls=(0, (3, 3))); ax.text(1.05, eps * 1.015, f"target ×{eps}", color=TINTA2, fontsize=7.5, va="bottom")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
ax.set_yticks([1, 1.2, 1.5, 2, 3, 5]); ax.set_yticklabels(["1", "1.2", "1.5", "2", "3", "5"]); ax.minorticks_off()
ax.set_xlabel("Mean simulated time per system (ns)"); ax.set_ylabel("Typical error factor vs. $D_{app}$(25 ns)")
ax.legend(frameon=False, fontsize=7, loc="upper right")
fig.savefig("paper/figures/fig6_cost_error.pdf"); fig.savefig("paper/figures/fig6_cost_error.png"); plt.close(fig)

# --- tabla y macros ---
NOM = {"fija-extremos": "Fixed length, end-point", "fija-ajuste": "Fixed length, linear fit", "fija-potencial": "Fixed length, power law",
       "regimen": "Regime rule, linear fit", "precision-ajuste": "Precision rule, linear fit", "precision-potencial": "Precision rule, power law"}
with open("paper/tabla_frontera.tex", "w") as fh:
    fh.write("% Generado por frontera.py. No editar a mano.\n\\begin{tabular}{l" + "rrr" * 2 + "}\n\\toprule\n")
    fh.write(" & \\multicolumn{3}{c}{Target error $\\times1.5$} & \\multicolumn{3}{c}{Target error $\\times1.3$} \\\\\n")
    fh.write("Rule & Time (ns) & Error & Target met & Time (ns) & Error & Target met \\\\\n\\midrule\n")
    for f in FAMILIAS:
        celdas = []
        for eps in (1.5, 1.3):
            x = reserva[f"{f}|{eps}"]
            celdas += (["n/a", "n/a", "0\\%"] if x["tiempo_test_mediana"] is None else
                       [f"{x['tiempo_test_mediana']:.1f}", f"$\\times${x['error_test_mediana']:.2f}", f"{100 * x['frac_cumple_test']:.0f}\\%"])
        fh.write(NOM[f] + " & " + " & ".join(celdas) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
M = {"frN": str(n), "frSplits": str(N_PART)}
nomf = {"fija-extremos": "FixEnd", "fija-ajuste": "FixFit", "fija-potencial": "FixPow", "regimen": "Regime",
        "precision-ajuste": "PrecFit", "precision-potencial": "PrecPow"}
for eps, ne in ((1.5, "OneFive"), (1.3, "OneThree"), (1.2, "OneTwo")):
    for f in FAMILIAS:
        m = todo["minimo"][f"{f}|{eps}"]; x = reserva[f"{f}|{eps}"]
        M[f"frMin{nomf[f]}{ne}"] = "n/a" if m is None else f"{m:.1f}"
        M[f"frTest{nomf[f]}{ne}"] = "n/a" if x["tiempo_test_mediana"] is None else f"{x['tiempo_test_mediana']:.1f}"
        M[f"frMet{nomf[f]}{ne}"] = f"{100 * x['frac_cumple_test']:.0f}"
with open("paper/numeros_frontera.tex", "w") as fh:
    fh.write("% Generado por frontera.py a partir de resultados/frontera.json. No editar a mano.\n")
    for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

print(f"{n} trayectorias; tiempo mínimo (ns) para alcanzar cada error típico con todas las corridas:")
for eps in OBJETIVOS:
    print(f"  ×{eps}: " + "  ".join(f"{f} {todo['minimo'][f'{f}|{eps}']}" for f in FAMILIAS))
print("elección en una mitad, evaluación en la otra (mediana de tiempo, error, % de particiones que cumplen):")
for eps in OBJETIVOS:
    print(f"  ×{eps}: " + "  ".join(f"{f} {reserva[f'{f}|{eps}']['tiempo_test_mediana']} / {reserva[f'{f}|{eps}']['error_test_mediana'] and round(reserva[f'{f}|{eps}']['error_test_mediana'], 2)} / {reserva[f'{f}|{eps}']['frac_cumple_test']:.2f}" for f in FAMILIAS))
