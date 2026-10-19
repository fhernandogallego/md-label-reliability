"""Sensibilidad de los resultados basados en curvas a las decisiones de análisis: ventana del ajuste del exponente,
umbrales de régimen y retardo de referencia. Genera resultados/sensibilidad.json y paper/tabla_sensibilidad.tex."""
import glob, json
import numpy as np
from scipy import stats

curvas = {}
for f in glob.glob("data/curvas/*.json"):
    d = json.load(open(f)); ns = d["condiciones"]["simulation_length"]
    t = np.array(d["li"]["time"], float); m = np.array(d["li"]["msd"], float)
    curvas[int(d["id"])] = (ns, t / t[-1] * ns, m)

def exponente(t, m, a, b):
    k = (t >= a * t[-1]) & (t <= b * t[-1]) & (m > 0)
    return np.polyfit(np.log(t[k]), np.log(m[k]), 1)[0]

R = {"ventanas": [], "referencias": []}
for a, b in ((0.1, 0.5), (0.2, 0.6), (0.3, 0.7), (0.4, 0.8)) if curvas else ():
    fila = {"ventana": [a, b]}
    for ns in (5.0, 50.0):
        be = np.array([exponente(t, m, a, b) for n, t, m in curvas.values() if n == ns]); e = f"{ns:g}"
        fila[e] = {"mediana": float(np.median(be)), "sub_0.6": float((be < 0.6).mean()), "sub_0.7": float((be < 0.7).mean()), "sub_0.8": float((be < 0.8).mean()),
                   "dif_0.8": float((be >= 0.8).mean()), "dif_0.9": float((be >= 0.9).mean()), "dif_0.95": float((be >= 0.95).mean())}
    R["ventanas"].append(fila)
largas = [(t, m) for n, t, m in curvas.values() if n == 50.0]
for ref in (15.0, 20.0, 25.0, 30.0, 35.0) if curvas else ():
    D = lambda tau: np.array([np.interp(tau, t, m) / (6 * tau) for t, m in largas])
    dref = D(ref); fila = {"ref_ns": ref}
    for tau in (1.0, 5.0, 10.0):
        fila[f"{tau:g}"] = {"razon_mediana": float(np.median(D(tau) / dref)), "spearman": float(stats.spearmanr(D(tau), dref)[0])}
    R["referencias"].append(fila)
if curvas: json.dump(R, open("resultados/sensibilidad.json", "w"), indent=1)
else: R = json.load(open("resultados/sensibilidad.json"))                           # sin data/curvas: regenerar tablas desde el JSON

pc = lambda x: f"{100 * x:.2f}" if 0 < x < 0.01 else f"{100 * x:.1f}"               # mismo formato que la tabla 2 del texto
with open("paper/tabla_sensibilidad.tex", "w") as fh:
    fh.write("% Generado por sensibilidad.py. No editar a mano.\n")
    fh.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{\\SI{5}{\\nano\\second} runs} & \\multicolumn{3}{c}{\\SI{50}{\\nano\\second} runs} \\\\\n"
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nFit window (fraction of run) & Median $\\beta$ & $\\beta<0.7$ & $\\beta\\ge0.9$ & Median $\\beta$ & $\\beta<0.7$ & $\\beta\\ge0.9$ \\\\\n\\midrule\n")
    for f in R["ventanas"]:
        a, b = f["ventana"]; s, l = f["5"], f["50"]
        fh.write(f"{a:.1f} to {b:.1f} & {s['mediana']:.2f} & {pc(s['sub_0.7'])}\\% & {pc(s['dif_0.9'])}\\% & {l['mediana']:.2f} & {pc(l['sub_0.7'])}\\% & {pc(l['dif_0.9'])}\\% \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
with open("paper/tabla_sensibilidad_ref.tex", "w") as fh:
    fh.write("% Generado por sensibilidad.py. No editar a mano.\n")
    fh.write("\\begin{tabular}{rrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Median $\\Dapp(\\tau)/\\Dapp(\\tau_{\\mathrm{ref}})$} & \\multicolumn{3}{c}{Spearman $\\rho$} \\\\\n"
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n$\\tau_{\\mathrm{ref}}$ (ns) & $\\tau=1$ & $\\tau=5$ & $\\tau=10$ & $\\tau=1$ & $\\tau=5$ & $\\tau=10$ \\\\\n\\midrule\n")
    for f in R["referencias"]:
        fh.write(f"{f['ref_ns']:g} & {f['1']['razon_mediana']:.2f} & {f['5']['razon_mediana']:.2f} & {f['10']['razon_mediana']:.2f} & "
                 f"{f['1']['spearman']:.2f} & {f['5']['spearman']:.2f} & {f['10']['spearman']:.2f} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
for f in R["ventanas"]:
    print("ventana", f["ventana"], "| 5 ns: mediana %.2f, <0,7 %.1f%%, ≥0,9 %.2f%% | 50 ns: mediana %.2f, <0,7 %.1f%%, ≥0,9 %.1f%%" % (
        f["5"]["mediana"], 100 * f["5"]["sub_0.7"], 100 * f["5"]["dif_0.9"], f["50"]["mediana"], 100 * f["50"]["sub_0.7"], 100 * f["50"]["dif_0.9"]))
for f in R["referencias"]:
    print("ref %g ns | razón a 1/5/10 ns: %.2f %.2f %.2f | Spearman: %.2f %.2f %.2f" % (f["ref_ns"], *[f[k]["razon_mediana"] for k in ("1", "5", "10")], *[f[k]["spearman"] for k in ("1", "5", "10")]))
