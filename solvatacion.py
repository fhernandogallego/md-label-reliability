"""Química de la solvatación del Li en las corridas largas y su relación con la fiabilidad de las etiquetas cortas.

Para cada corrida larga (data/iones/<id>.npz, de extraer_iones.py: Li, N del TFSI y átomos del polímero que pueden solvatar
al Li, según polymer_solvate_types de HTP-MD; coordenadas desenvueltas cada 2 ps):
  - coordinación por el polímero: nº medio de átomos del polímero a menos de R_POL = 3,4 Å del Li (el mismo radio que usa HTP-MD para Li–O(TFSI), elegido de su g(r), y mínimo de la g(r) Li–polímero medida aquí; con 2,6 Å la capa queda cortada con PCFF+)
    y fracción de Li sin ningún átomo del polímero en esa capa;
  - persistencia de la solvatación: función de correlación intermitente de los contactos Li–átomo del polímero,
    C(τ) = <c(t)c(t+τ)>/<c>, a 1 y 5 ns (fracción de contactos que siguen; el tiempo a 1/e se guarda, pero muchas veces
    supera la corrida y queda censurado);
  - emparejamiento iónico: de la matriz de agregados de HTP-MD (population.txt, Li–O(TFSI) ≤ 3,4 Å, nº medio de agregados
    con i Li y j TFSI), fracción de Li libre (agregado de un Li sin anión) y fracción de Li en agregados de 5 Li o más.
Después, correlación de rangos (con intervalo bootstrap) de cada descriptor con: ln D a 50 ns, exponente efectivo β a 50 ns,
sobreestimación de la corrida corta (ln D5/D50, mismo sistema), error de orden (|rango corto − rango largo|) y ruido entre
réplicas (|ln A − ln B|, sistemas replicados); para el ruido, también la correlación parcial descontando ln D50, porque el
ruido depende de la movilidad.
Uso: .venv/bin/python solvatacion.py → resultados/solvatacion.json, resultados/solvatacion_por_sistema.csv, paper/numeros_solvatacion.tex
(caché por corrida en resultados/solvatacion/<id>.json: solo calcula las nuevas)
"""
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats
from comun import N_BOOT, SEMILLA_BOOT, ic

R_POL, PASO, LAGS_PS = 3.4, 5, (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
os.makedirs("resultados/solvatacion", exist_ok=True)


def describir(f):
    d = np.load(f); L = d["caja"]; dt = float(d["dt_ps"]) * PASO
    li, po = d["li"][::PASO].astype(np.float32), d["polimero"][::PASO].astype(np.float32)
    T = len(li); lags = [l for l in LAGS_PS if l // dt < T // 2]
    cn, sin, num, den = [], [], np.zeros(len(lags)), np.zeros(len(lags))
    for k in range(li.shape[1]):
        r = po - li[:, k:k + 1, :]; r -= L * np.round(r / L)
        c = (np.einsum("tjx,tjx->tj", r, r) < R_POL ** 2)                      # contactos (tiempo × átomo)
        n = c.sum(1); cn.append(n.mean()); sin.append((n == 0).mean())
        for i, lag in enumerate(lags):
            s = int(lag // dt); num[i] += (c[:-s] & c[s:]).sum(); den[i] += c[:-s].sum()
    C = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    debajo = np.where(C < np.exp(-1))[0]
    if len(debajo):                                                          # interpolación en log(t)
        j = debajo[0]; t0, t1 = (np.log(lags[j - 1]), np.log(lags[j])) if j else (np.log(dt), np.log(lags[0]))
        c0 = C[j - 1] if j else 1.0
        tres, cens = float(np.exp(t0 + (c0 - np.exp(-1)) / (c0 - C[j]) * (t1 - t0))), False
    else:
        tres, cens = float(lags[-1]), True
    P = d["poblacion"]; i_, j_ = np.indices(P.shape); nli = (i_ * P).sum()
    return {"cn_polimero": float(np.mean(cn)), "frac_li_sin_polimero": float(np.mean(sin)),
            "residencia_ps": tres, "residencia_censurada": cens, "C_residencia": {str(l): float(x) for l, x in zip(lags, C)},
            "frac_li_libre": float(P[1, 0] / nli) if P.shape[0] > 1 else np.nan,
            "frac_li_agregado_grande": float((i_ * P)[5:].sum() / nli) if nli else np.nan,
            "persistencia_1ns": float(C[lags.index(1000)]), "persistencia_5ns": float(C[lags.index(5000)])}


T = pd.read_csv("resultados/tabla_por_trayectoria.csv")
L50 = T[T.ns == 50].set_index("id")
filas = []
ids = {int(os.path.basename(f)[:-4]) for f in glob.glob("data/iones/*.npz") if "parcial" not in f}
ids |= {int(os.path.basename(f)[:-5]) for f in glob.glob("resultados/solvatacion/*.json")}   # sin data/: solo cachés
for i in sorted(ids):
    if i not in L50.index: continue
    cache = f"resultados/solvatacion/{i}.json"
    if not os.path.exists(cache): json.dump(describir(f"data/iones/{i}.npz"), open(cache, "w"))
    filas.append({"id": i, "sistema": L50.loc[i, "sistema"], **{k: v for k, v in json.load(open(cache)).items() if k != "C_residencia"}})
S = pd.DataFrame(filas).set_index("sistema")

# objetivos por sistema
c = T[T.ns == 5].sort_values("id"); orig = c[~c.carpeta.str.contains("new_config", na=False)].drop_duplicates("sistema").set_index("sistema")
S["literatura"] = L50.set_index("sistema").loc[S.index, "literatura"].astype(bool)
S["lnD50"] = np.log(L50.set_index("sistema").loc[S.index, "D_medio"])
S["beta50"] = L50.set_index("sistema").loc[S.index, "beta_medio"]
S["lnD5"] = np.log(orig.loc[S.index, "D_extremos"])
S["sobreestimacion"] = S.lnD5 - S.lnD50
S["error_orden"] = (S.lnD5.rank() - S.lnD50.rank()).abs() / len(S)
rep = c[c.sistema.isin(S.index)].groupby("sistema").filter(lambda g: len(g) == 2).sort_values(["sistema", "id"])
S["ruido_replicas"] = rep.groupby("sistema").D_extremos.apply(lambda x: abs(np.log(x.iloc[0]) - np.log(x.iloc[1])))
S.to_csv("resultados/solvatacion_por_sistema.csv")

DESC = ["cn_polimero", "frac_li_sin_polimero", "persistencia_1ns", "persistencia_5ns", "frac_li_libre", "frac_li_agregado_grande"]
OBJ = ["lnD50", "beta50", "sobreestimacion", "error_orden", "ruido_replicas"]
rng = np.random.default_rng(SEMILLA_BOOT)


def rho(a, b): return stats.spearmanr(a, b)[0]


def parcial(a, b, z):
    """Spearman parcial de a y b descontando z (sobre rangos)."""
    ra, rb, rz = (pd.Series(v).rank().values for v in (a, b, z))
    ea = ra - np.polyval(np.polyfit(rz, ra, 1), rz); eb = rb - np.polyval(np.polyfit(rz, rb, 1), rz)
    return np.corrcoef(ea, eb)[0, 1]


R = {"n": len(S), "n_cribado": int((~S.literatura).sum()), "r_pol_A": R_POL, "residencia_censuradas": int(S.residencia_censurada.sum()),
     "medianas": {k: float(S[k].median()) for k in DESC}, "medianas_cribado_literatura": {
         k: [float(S[~S.literatura][k].median()), float(S[S.literatura][k].median())] for k in DESC}, "correlaciones": {}}
for dsc in DESC:
    for obj in OBJ:
        Q = S[list(dict.fromkeys([dsc, obj, "lnD50"]))].dropna(); a, b, z = Q[dsc].values, Q[obj].values, Q["lnD50"].values
        bs = [rho(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(N_BOOT))]
        e = {"n": len(Q), "spearman": float(rho(a, b)), "ic95": ic(bs)}
        if obj in ("ruido_replicas", "sobreestimacion", "error_orden", "beta50"):
            bp = [parcial(a[i], b[i], z[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(N_BOOT))]
            e.update({"parcial_sin_lnD50": float(parcial(a, b, z)), "parcial_ic95": ic(bp)})
        R["correlaciones"][f"{dsc}|{obj}"] = e
json.dump(R, open("resultados/solvatacion.json", "w"), indent=1)

print(f"{len(S)} corridas largas ({R['n_cribado']} de cribado); residencia censurada en {R['residencia_censuradas']}")
print("medianas (cribado | literatura):", {k: [round(v, 3) for v in x] for k, x in R["medianas_cribado_literatura"].items()})
for k, e in R["correlaciones"].items():
    extra = f" | parcial sin lnD50 {e['parcial_sin_lnD50']:.2f} [{e['parcial_ic95'][0]:.2f}, {e['parcial_ic95'][1]:.2f}]" if "parcial_sin_lnD50" in e else ""
    print(f"{k:40s} n={e['n']:3d} ρ={e['spearman']:+.2f} [{e['ic95'][0]:+.2f}, {e['ic95'][1]:+.2f}]{extra}")

# ---------- Macros y tabla de la información suplementaria ----------
f2 = lambda v: f"{v:.2f}"; pc = lambda v: f"{100 * v:.0f}"

M = {"solvN": str(R["n"]), "solvNScreen": str(R["n_cribado"]), "solvNLit": str(R["n"] - R["n_cribado"]), "solvRcut": f"{R_POL:.1f}"}
for k, nom, fmt in (("cn_polimero", "Cn", lambda v: f"{v:.1f}"), ("frac_li_sin_polimero", "NoPol", pc), ("persistencia_1ns", "PersOne", pc),
                    ("persistencia_5ns", "PersFive", pc), ("frac_li_libre", "Free", pc), ("frac_li_agregado_grande", "Aggr", pc)):
    a, b = R["medianas_cribado_literatura"][k]; M[f"solv{nom}Screen"], M[f"solv{nom}Lit"] = fmt(a), fmt(b)
for k, nom in (("frac_li_agregado_grande|lnD50", "AggrD"), ("frac_li_agregado_grande|sobreestimacion", "AggrOver"),
               ("frac_li_agregado_grande|error_orden", "AggrRank"), ("frac_li_agregado_grande|ruido_replicas", "AggrNoise")):
    e = R["correlaciones"][k]; M[f"solv{nom}"] = f2(e["spearman"]); M[f"solv{nom}Lo"], M[f"solv{nom}Hi"] = (f2(v) for v in e["ic95"])
ruido = [R["correlaciones"][f"{d}|ruido_replicas"] for d in DESC]
M["solvNoiseMaxAbs"] = f2(max(abs(e["spearman"]) for e in ruido))
with open("paper/numeros_solvatacion.tex", "w") as fh:
    fh.write("% Generado por solvatacion.py a partir de resultados/solvatacion.json. No editar a mano.\n")
    for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
NOMD = {"cn_polimero": "Polymer atoms in the first shell", "frac_li_sin_polimero": "Li$^+$ without polymer contact",
        "persistencia_1ns": "Polymer contacts kept after 1 ns", "persistencia_5ns": "Polymer contacts kept after 5 ns",
        "frac_li_libre": "Free Li$^+$ (no anion)", "frac_li_agregado_grande": "Li$^+$ in clusters of $\\geq5$ Li$^+$"}
NOMO = {"lnD50": "$\\ln D$, 50 ns", "sobreestimacion": "Overestimation", "error_orden": "Rank error", "ruido_replicas": "Replica noise"}
with open("paper/tabla_solvatacion.tex", "w") as fh:
    fh.write("% Generado por solvatacion.py. No editar a mano.\n\\begin{tabular}{lrr" + "r" * len(NOMO) + "}\n\\toprule\n"
             " & \\multicolumn{2}{c}{Median} & \\multicolumn{" + str(len(NOMO)) + "}{c}{Rank correlation with} \\\\\n"
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-" + str(3 + len(NOMO)) + "}\nDescriptor & Screening & Literature & " + " & ".join(NOMO.values()) + " \\\\\n\\midrule\n")
    for k, nom in NOMD.items():
        a, b = R["medianas_cribado_literatura"][k]; fa = (lambda v: f"{v:.1f}") if k == "cn_polimero" else (lambda v: f"{100 * v:.0f}\\%")
        celdas = []
        for o in NOMO:
            e = R["correlaciones"][f"{k}|{o}"]; sig = e["ic95"][0] > 0 or e["ic95"][1] < 0
            celdas.append(("\\textbf{" if sig else "") + f"{e['spearman']:+.2f}" + ("}" if sig else ""))
        fh.write(f"{nom} & {fa(a)} & {fa(b)} & " + " & ".join(celdas) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
