"""Techo de ruido de las etiquetas de 5 ns: qué puede aprender un modelo de ellas y cuánto cuesta una etiqueta fiable.

Lee resultados/tabla_por_trayectoria.csv (generada por analizar_curvas.py), así que no necesita data/.
Todo en escala logarítmica, ln D. Modelo de medida: etiqueta = valor del sistema + ruido de la corrida (σ_run, de las réplicas).
  - Techo: un predictor perfecto del valor del sistema correlaciona con UNA etiqueta de 5 ns como mucho √r (Pearson) y explica
    como mucho R² = r, con r la fiabilidad de una corrida (misma definición que el texto: 1 − σ_run²/σ_tot²).
  - Heterocedasticidad: ¿depende el ruido de la movilidad? (|diferencia entre réplicas| frente al nivel medio).
  - Spearman–Brown: fiabilidad de la media de n réplicas, r_n = n·r / (1 + (n−1)·r), y réplicas necesarias para r ≥ 0,8.
  - Origen del ruido: las réplicas parten de estructuras amorfas distintas (ruido de estructura + de dinámica), mientras que la
    corrida de 5 ns de un par y los primeros 5 ns de su corrida larga parten de la MISMA estructura inicial (Xie et al. 2022:
    "computed from the same random initial structures"; aquí se comprueba: razón mediana 1,00). Su diferencia mide solo el
    ruido de dinámica, y σ_estructura² = σ_run² − σ_dinámica². Por la misma razón los pares NO son independientes y no se
    corrige su correlación por fiabilidad.
  - Recall máximo: simulación gaussiana con la varianza entre sistemas y el ruido medidos; proporción del verdadero top-k
    que recupera la selección por la media de n corridas de 5 ns (equivale al recall aparente de un modelo perfecto
    evaluado contra esas etiquetas).
Intervalos: bootstrap por sistemas (réplicas, conjunto de cortas y pares remuestreados de forma independiente).
Uso: python techo.py → resultados/techo.json, paper/tabla_techo.tex, paper/numeros_techo.tex
"""
import glob, json
import numpy as np
import pandas as pd
from scipy import stats

N_BOOT, N_SIM, SEMILLA = 2000, 200_000, 20261016
REPLICAS = (1, 2, 3, 4)
TOPS = (0.05, 0.10, 0.20)
rng = np.random.default_rng(SEMILLA)
T = pd.read_csv("resultados/tabla_por_trayectoria.csv")
cortas = T[T.ns == 5].sort_values("id")
ic = lambda b: [float(np.nanquantile(b, .025)), float(np.nanquantile(b, .975))]
rep = cortas.groupby("sistema").filter(lambda g: len(g) == 2).sort_values(["sistema", "id"])
a = cortas.groupby("sistema").agg(D5_extremos=("D_extremos", "first"), D5_medio=("D_medio", "first"), literatura=("literatura", "first"))
P = a.join(T[T.ns == 50].set_index("sistema")[["D_medio"]].rename(columns={"D_medio": "D50"}), how="inner")
pref = {}                                                         # primeros 5 ns de cada corrida larga (cachés de analizar_prefijos.py)
for f in glob.glob("resultados/prefijos/*.json"):
    r = json.load(open(f))
    if r["ns"] >= 49: pref[int(f.split("/")[-1][:-5])] = r["prefijos"]["5"]
L = T[T.ns == 50].set_index("sistema")
P["pref_extremos"] = [pref[int(L.loc[s, "id"])]["extremos"] if int(L.loc[s, "id"]) in pref else np.nan for s in P.index]
P["pref_medio"] = [pref[int(L.loc[s, "id"])]["mitad"] if int(L.loc[s, "id"]) in pref else np.nan for s in P.index]
R = {"n_boot": N_BOOT, "n_sim": N_SIM, "semilla": SEMILLA, "n_replicados": int(rep.sistema.nunique()),
     "n_pares_cribado": int((~P.literatura).sum()), "n_pares": len(P)}


def spearman_brown(r, n):
    return n * r / (1 + (n - 1) * r)


def recall_simulado(r, n, frac, rng):
    """Proporción del verdadero top-frac recuperada al seleccionar por la media de n corridas (varianza total 1)."""
    v = rng.standard_normal(N_SIM) * np.sqrt(r)
    obs = v + rng.standard_normal(N_SIM) * np.sqrt((1 - r) / n)
    k = int(frac * N_SIM)
    return len(np.intersect1d(np.argpartition(-v, k)[:k], np.argpartition(-obs, k)[:k], assume_unique=True)) / k


def medidas(x, y, todas, ps, pp):
    """x, y: ln D de las dos réplicas; todas: ln D de todas las cortas; ps, pp: ln D de la corrida de 5 ns de cada par de
    cribado y de los primeros 5 ns de su corrida larga (misma estructura inicial)."""
    s_run = (x - y).std(ddof=1) / np.sqrt(2)
    r = 1 - s_run ** 2 / todas.var(ddof=1)
    s_din = (ps - pp).std(ddof=1) / np.sqrt(2)
    s_est2 = max(s_run ** 2 - s_din ** 2, 0)
    return {"sd_run": s_run, "fiabilidad": r, "techo_correlacion": np.sqrt(max(r, 0)), "techo_r2": r,
            "sd_dinamica": s_din, "sd_estructura": np.sqrt(s_est2), "frac_var_estructura": s_est2 / s_run ** 2,
            "razon_mediana_corta_prefijo": np.exp(np.median(ps - pp)),
            **{f"fiabilidad_{n}rep": spearman_brown(r, n) for n in REPLICAS[1:]},
            "replicas_para_0.8": 0.8 * (1 - r) / (0.2 * r)}


cribado = P[~P.literatura]
for col, col_par, col_pref in (("D_extremos", "D5_extremos", "pref_extremos"), ("D_medio", "D5_medio", "pref_medio")):
    x = np.log(rep.groupby("sistema")[col].first().values); y = np.log(rep.groupby("sistema")[col].last().values)
    c = cribado.dropna(subset=[col_pref])
    todas = np.log(cortas[col].values); ps = np.log(c[col_par].values); pp = np.log(c[col_pref].values)
    m = medidas(x, y, todas, ps, pp); boot = {k: [] for k in m}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x)); j = rng.integers(0, len(todas), len(todas)); p = rng.integers(0, len(ps), len(ps))
        for k, v in medidas(x[i], y[i], todas[j], ps[p], pp[p]).items(): boot[k].append(v)
    e = {k: float(v) for k, v in m.items()}
    e.update({f"{k}_ic95": ic(boot[k]) for k in m})
    U = json.load(open("resultados/incertidumbre.json"))["replicas"][col]  # mismo intervalo de r que el resto del texto
    e["fiabilidad_ic95"] = U["fiabilidad_ic95"]; e["techo_correlacion_ic95"] = [float(np.sqrt(max(v, 0))) for v in U["fiabilidad_ic95"]]
    nivel = (x + y) / 2; dif = np.abs(x - y)                       # heterocedasticidad
    mitad = nivel < np.median(nivel)
    e["heterocedasticidad"] = {"spearman_absdif_vs_nivel": float(stats.spearmanr(dif, nivel)[0]),
                               "p": float(stats.spearmanr(dif, nivel)[1]),
                               "sd_run_lentos": float((x - y)[mitad].std(ddof=1) / np.sqrt(2)),
                               "sd_run_rapidos": float((x - y)[~mitad].std(ddof=1) / np.sqrt(2))}
    e["recall"] = {}
    for n in REPLICAS:
        for frac in TOPS:
            sim = np.random.default_rng(SEMILLA + n)
            fila = {"punto": recall_simulado(m["fiabilidad"], n, frac, sim)}
            fila["ic95"] = [recall_simulado(max(r_, 1e-3), n, frac, sim) for r_ in e["fiabilidad_ic95"]]
            e["recall"][f"{n}rep_top{round(100 * frac)}"] = fila
    R[col] = e

json.dump(R, open("resultados/techo.json", "w"), indent=1)

f2 = lambda v: f"{v:.2f}"
NOM = {"D_extremos": "End-point", "D_medio": "Origin-averaged, lag $T/2$"}
with open("paper/tabla_techo.tex", "w") as fh:
    fh.write("% Generado por techo.py. No editar a mano.\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
             " & & & & \\multicolumn{3}{c}{Recall of the true top} \\\\\n\\cmidrule(lr){5-7}\n"
             "Label & Runs averaged & Reliability & Max.\\ correlation & 5\\% & 10\\% & 20\\% \\\\\n\\midrule\n")
    for col in ("D_extremos", "D_medio"):
        e = R[col]
        for j, n in enumerate(REPLICAS):
            r = e["fiabilidad"] if n == 1 else e[f"fiabilidad_{n}rep"]
            rec = " & ".join(f"{100 * e['recall'][f'{n}rep_top{t}']['punto']:.0f}\\%" for t in (5, 10, 20))
            fh.write(f"{NOM[col] if j == 0 else ''} & {n} & {f2(r)} & {f2(np.sqrt(r))} & {rec} \\\\\n")
        if col == "D_extremos": fh.write("\\addlinespace\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

M = {"techoNRep": str(R["n_replicados"]), "techoNPairsScreen": str(R["n_pares_cribado"])}
for col, nom in (("D_extremos", "End"), ("D_medio", "Avg")):
    e = R[col]
    M[f"techo{nom}Corr"] = f2(e["techo_correlacion"]); M[f"techo{nom}CorrLo"], M[f"techo{nom}CorrHi"] = (f2(v) for v in e["techo_correlacion_ic95"])
    M[f"techo{nom}RTwo"] = f2(e["techo_r2"])
    M[f"techo{nom}RelTwo"] = f2(e["fiabilidad_2rep"]); M[f"techo{nom}RelFour"] = f2(e["fiabilidad_4rep"])
    M[f"techo{nom}RepForEight"] = f"{np.ceil(e['replicas_para_0.8']):.0f}"
    M[f"techo{nom}SdDyn"] = f2(e["sd_dinamica"]); M[f"techo{nom}SdStruct"] = f2(e["sd_estructura"])
    M[f"techo{nom}FracStructPct"] = f"{100 * e['frac_var_estructura']:.0f}"
    M[f"techo{nom}FracStructLo"], M[f"techo{nom}FracStructHi"] = (f"{100 * v:.0f}" for v in e["frac_var_estructura_ic95"])
    M[f"techo{nom}ShortPrefRatio"] = f2(e["razon_mediana_corta_prefijo"])
    h = e["heterocedasticidad"]
    M[f"techo{nom}HetRho"] = f2(h["spearman_absdif_vs_nivel"]); M[f"techo{nom}HetP"] = f2(h["p"])
    M[f"techo{nom}SdSlow"] = f2(h["sd_run_lentos"]); M[f"techo{nom}SdFast"] = f2(h["sd_run_rapidos"])
    for n, nn in ((1, "One"), (2, "Two"), (4, "Four")):
        for t, tn in ((5, "Five"), (10, "Ten"), (20, "Twenty")):
            M[f"techo{nom}Recall{nn}Top{tn}"] = f"{100 * e['recall'][f'{n}rep_top{t}']['punto']:.0f}"
    lo, hi = e["recall"]["1rep_top10"]["ic95"]; M[f"techo{nom}RecallOneTopTenLo"] = f"{100 * lo:.0f}"; M[f"techo{nom}RecallOneTopTenHi"] = f"{100 * hi:.0f}"
with open("paper/numeros_techo.tex", "w") as fh:
    fh.write("% Generado por techo.py a partir de resultados/techo.json. No editar a mano.\n")
    for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

for col in ("D_extremos", "D_medio"):
    e = R[col]; h = e["heterocedasticidad"]
    print(f"\n{col}: r = {e['fiabilidad']:.2f} {ic(np.array(e['fiabilidad_ic95']))} → techo de correlación {e['techo_correlacion']:.2f} "
          f"[{e['techo_correlacion_ic95'][0]:.2f}–{e['techo_correlacion_ic95'][1]:.2f}], R² ≤ {e['techo_r2']:.2f}")
    print(f"   réplicas: r2 = {e['fiabilidad_2rep']:.2f}, r4 = {e['fiabilidad_4rep']:.2f}, para r ≥ 0,8 hacen falta {e['replicas_para_0.8']:.1f}")
    print(f"   ruido: dinámica {e['sd_dinamica']:.2f}, estructura {e['sd_estructura']:.2f} → {e['frac_var_estructura']:.0%} de la varianza es estructura "
          f"[{e['frac_var_estructura_ic95'][0]:.0%}–{e['frac_var_estructura_ic95'][1]:.0%}]; razón mediana corta/prefijo {e['razon_mediana_corta_prefijo']:.2f}")
    print(f"   heterocedasticidad: ρ(|dif|, nivel) = {h['spearman_absdif_vs_nivel']:.2f} (p = {h['p']:.2f}); σ_run lentos {h['sd_run_lentos']:.2f}, rápidos {h['sd_run_rapidos']:.2f}")
    print("   recall top 5/10/20 %:", {n: [round(e['recall'][f'{n}rep_top{t}']['punto'], 2) for t in (5, 10, 20)] for n in REPLICAS})
