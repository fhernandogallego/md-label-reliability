"""Demostración de aprendizaje automático: ¿cómo se comporta un modelo real frente al techo de ruido de las etiquetas de 5 ns?

Modelo: random forest sobre la huella Morgan (radio 2, 2048 bits con recuento) del monómero (mol_smiles de HTP-MD, con los
átomos ficticios de los extremos). Etiqueta de entrenamiento: ln D end-point de la primera corrida de 5 ns de cada sistema
(la corrida original del cribado), que es la etiqueta que distribuye HTP-MD. Validación cruzada en 5 particiones DIVIDIENDO POR
SISTEMA: todas las predicciones son fuera de muestra.
Preguntas:
  1. Correlación (Pearson, ln D) de la predicción con una etiqueta de 5 ns, frente al techo √r de techo.py; la correlación
     implícita con la propiedad del sistema es r_obs/√r.
  2. En los sistemas replicados: correlación con la etiqueta de cada corrida y con la media de las dos; el modelo de medida
     predice que la segunda es mayor en un factor √(r₂/r), con r₂ la fiabilidad de la media de dos corridas.
  3. En los sistemas con corrida de 50 ns: ¿ordena el modelo la referencia larga mejor que la propia corrida corta del
     sistema? (Xie et al. 2022 afirman que el modelo supera a una simulación individual.) Se separan cribado y literatura.
  4. Fuga: la misma validación dividiendo por TRAYECTORIA (las dos réplicas de un sistema pueden caer una en entrenamiento y
     otra en prueba); se compara, en las réplicas, la correlación con su propia etiqueta frente a la división por sistema.
  5. Sensibilidad a la química: validación cruzada por familias de monómeros (Butina sobre Tanimoto, dos cortes), para ver
     si la correlación sigue cerca del techo cuando el modelo no ha visto monómeros parecidos.
Intervalos: bootstrap por sistemas (2000 remuestreos). Necesita rdkit y scikit-learn (entorno .venv del proyecto).
Uso: .venv/bin/python ml_demo.py → resultados/ml_demo.json, paper/numeros_ml.tex, paper/tabla_ml.tex
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem, DataStructs, RDLogger
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold

RDLogger.DisableLog("rdApp.*")
N_PART, N_BOOT, SEMILLA, N_BITS = 5, 2000, 20261019, 2048
rng = np.random.default_rng(SEMILLA)
T = pd.read_csv("resultados/tabla_por_trayectoria.csv")
E = pd.read_csv("resultados/estructuras.csv")[["id", "mol_smiles"]]
T = T.merge(E, on="id", how="left")
U = json.load(open("resultados/incertidumbre.json"))["replicas"]["D_extremos"]
H = json.load(open("resultados/techo.json"))["D_extremos"]
r1, r2 = U["fiabilidad"], H["fiabilidad_2rep"]

# ---------- Huellas ----------
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)
cache = {}


def huella(smi):
    if smi not in cache:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        cache[smi] = gen.GetCountFingerprintAsNumPy(m).astype(np.float32) if m is not None else None
    return cache[smi]


cortas = T[T.ns == 5].sort_values("id").copy()
cortas["X"] = cortas.mol_smiles.map(huella)
fallos = int(cortas.X.isna().sum()); cortas = cortas[cortas.X.notna()]
cortas["y"] = np.log(cortas.D_extremos)
cortas["original"] = ~cortas.carpeta.str.contains("new_config", na=False)
sis = cortas[cortas.original].drop_duplicates("sistema").set_index("sistema")      # un sistema = su corrida original
R = {"n_sistemas": len(sis), "n_trayectorias": len(cortas), "fallos_smiles": fallos, "fiabilidad": r1, "fiabilidad_2rep": r2,
     "techo_correlacion": float(np.sqrt(r1)), "particiones": N_PART}


def modelo():
    return RandomForestRegressor(n_estimators=400, max_features=0.3, min_samples_leaf=2, n_jobs=-1, random_state=SEMILLA)


def fuera_de_muestra(X, y, grupos):
    """Predicciones fuera de muestra con KFold sobre los grupos indicados (sistemas o trayectorias)."""
    pred = np.empty(len(y)); unicos = np.unique(grupos)
    for tr_g, te_g in KFold(N_PART, shuffle=True, random_state=SEMILLA).split(unicos):
        tr, te = np.isin(grupos, unicos[tr_g]), np.isin(grupos, unicos[te_g])
        pred[te] = modelo().fit(X[tr], y[tr]).predict(X[te])
    return pred


def r_ic(a, b):
    a, b = np.asarray(a), np.asarray(b); n = len(a)
    bs = [np.corrcoef(a[i], b[i])[0, 1] for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return float(np.corrcoef(a, b)[0, 1]), [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


# ---------- 1) División por sistema: todas las etiquetas ----------
X = np.stack(sis.X.values); y = sis.y.values
pred = pd.Series(fuera_de_muestra(X, y, np.arange(len(y))), index=sis.index)
c, ci = r_ic(pred.values, y)
R["todas"] = {"pearson": c, "pearson_ic95": ci, "r2": float(1 - np.mean((pred.values - y) ** 2) / np.var(y)),
              "fraccion_del_techo": c / np.sqrt(r1), "correlacion_implicita_con_propiedad": c / np.sqrt(r1),
              "correlacion_implicita_ic95": [v / np.sqrt(r1) for v in ci]}

# ---------- 2) Sistemas replicados: una etiqueta frente a la media de dos ----------
rep = cortas.groupby("sistema").filter(lambda g: len(g) == 2)
A = rep[rep.original].set_index("sistema").y; B = rep[~rep.original].set_index("sistema").y
s2 = [s for s in A.index if s in B.index and s in pred.index]
pa, ya, yb = pred[s2].values, A[s2].values, B[s2].values
R["replicados"] = {"n": len(s2)}
for nombre, obj in (("A", ya), ("B", yb), ("media", (ya + yb) / 2)):
    c_, ci_ = r_ic(pa, obj); R["replicados"][f"pearson_{nombre}"] = c_; R["replicados"][f"pearson_{nombre}_ic95"] = ci_
R["replicados"]["media_esperada_por_el_modelo_de_medida"] = (R["replicados"]["pearson_A"] + R["replicados"]["pearson_B"]) / 2 * np.sqrt(r2 / r1)
boot = []
for _ in range(N_BOOT):
    i = rng.integers(0, len(s2), len(s2))
    boot.append(np.corrcoef(pa[i], (ya[i] + yb[i]) / 2)[0, 1] - (np.corrcoef(pa[i], ya[i])[0, 1] + np.corrcoef(pa[i], yb[i])[0, 1]) / 2)
R["replicados"]["ganancia_media_ic95"] = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]

# ---------- 3) Referencia larga: modelo frente a la propia corrida corta ----------
L = T[T.ns == 50].set_index("sistema")
s3 = [s for s in L.index if s in pred.index]
ref = np.log(L.loc[s3, "D_medio"].values); corta = sis.loc[s3, "y"].values; pm = pred[s3].values; lit = L.loc[s3, "literatura"].values.astype(bool)
R["larga"] = {"n": len(s3)}
for nombre, sel in (("todos", np.ones(len(s3), bool)), ("cribado", ~lit), ("literatura", lit)):
    e = {"n": int(sel.sum())}
    for q, v in (("modelo", pm), ("corrida_corta", corta)):
        e[f"spearman_{q}"] = float(stats.spearmanr(v[sel], ref[sel])[0]); e[f"pearson_{q}"] = float(np.corrcoef(v[sel], ref[sel])[0, 1])
    d = []
    idx = np.where(sel)[0]
    for _ in range(N_BOOT):
        j = rng.choice(idx, len(idx))
        d.append(stats.spearmanr(pm[j], ref[j])[0] - stats.spearmanr(corta[j], ref[j])[0])
    e["dif_spearman_modelo_menos_corta_ic95"] = [float(np.quantile(d, .025)), float(np.quantile(d, .975))]
    R["larga"][nombre] = e

# ---------- 4) Fuga: división por trayectoria ----------
Xt = np.stack(cortas.X.values); yt = cortas.y.values
pred_t = fuera_de_muestra(Xt, yt, np.arange(len(yt)))                       # cada trayectoria es su propio grupo
cortas["pred_traj"] = pred_t
en_rep = cortas.sistema.isin(s2).values
sistema_por_tray = cortas.sistema.values; pliego = np.empty(len(yt), int)
for k, (_, te) in enumerate(KFold(N_PART, shuffle=True, random_state=SEMILLA).split(np.arange(len(yt)))): pliego[te] = k
compa = pd.Series(pliego, index=cortas.index).groupby(sistema_por_tray).transform("nunique").values > 1   # réplicas en pliegos distintos
sel = en_rep & compa
pred_sis_por_tray = pred.reindex(cortas.sistema).values
R["fuga"] = {"n_trayectorias_replicadas_con_pareja_en_entrenamiento": int(sel.sum()),
             "pearson_division_por_trayectoria": float(np.corrcoef(pred_t[sel], yt[sel])[0, 1]),
             "pearson_division_por_sistema": float(np.corrcoef(pred_sis_por_tray[sel], yt[sel])[0, 1]),
             "pearson_todas_division_por_trayectoria": float(np.corrcoef(pred_t, yt)[0, 1])}

# ---------- 5) Sensibilidad: validación cruzada por familias químicas ----------
# Con la división por sistema, un monómero de prueba suele tener en entrenamiento un vecino muy parecido. Aquí los monómeros
# se agrupan por similitud (Butina sobre la distancia de Tanimoto de la huella Morgan binaria, radio 2, 2048 bits) y cada
# familia entera cae en un solo pliego (GroupKFold, pliegos de tamaño equilibrado). Dos cortes: familias finas y gruesas.
# Va al final para no alterar los números aleatorios de lo anterior.
fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in sis.mol_smiles]
SIM = np.stack([np.asarray(DataStructs.BulkTanimotoSimilarity(f, fps), np.float32) for f in fps])


def vecino_en_entrenamiento(pliego):
    """Mediana, sobre los sistemas, de la mayor similitud de Tanimoto con un monómero de los pliegos de entrenamiento."""
    return float(np.median([SIM[i, pliego != pliego[i]].max() for i in range(len(pliego))]))


def pliegos(divisor, grupos):
    p = np.empty(len(grupos), int)
    for k, (_, te) in enumerate(divisor.split(np.zeros(len(grupos)), groups=grupos)): p[te] = k
    return p


p_sis = np.empty(len(y), int)
for k, (_, te) in enumerate(KFold(N_PART, shuffle=True, random_state=SEMILLA).split(y)): p_sis[te] = k
R["familias"] = {"vecino_division_por_sistema": vecino_en_entrenamiento(p_sis)}
s3c = [s for s, l in zip(s3, lit) if not l]
for nombre, corte in (("finas", 0.4), ("gruesas", 0.6)):
    d = [1 - float(SIM[i, j]) for i in range(1, len(fps)) for j in range(i)]
    grupos = np.empty(len(fps), int)
    for g, miembros in enumerate(Butina.ClusterData(d, len(fps), corte, isDistData=True, reordering=True)): grupos[list(miembros)] = g
    del d
    pl = pliegos(GroupKFold(N_PART, shuffle=True, random_state=SEMILLA), grupos)
    pf = np.empty(len(y))
    for k in range(N_PART): pf[pl == k] = modelo().fit(X[pl != k], y[pl != k]).predict(X[pl == k])
    pf = pd.Series(pf, index=sis.index)
    cf, cif = r_ic(pf.values, y); tam = np.bincount(grupos)
    R["familias"][nombre] = {"distancia_corte": corte, "similitud_corte": 1 - corte, "n_grupos": int(len(tam)),
                             "mayor_grupo_frac": float(tam.max() / len(y)), "unitarios": int((tam == 1).sum()),
                             "pearson": cf, "pearson_ic95": cif, "fraccion_del_techo": cf / np.sqrt(r1),
                             "vecino_en_entrenamiento": vecino_en_entrenamiento(pl),
                             "spearman_larga_cribado": float(stats.spearmanr(pf[s3c].values, np.log(L.loc[s3c, "D_medio"].values))[0])}

json.dump(R, open("resultados/ml_demo.json", "w"), indent=1, default=float)

# ---------- Macros y tabla ----------
f2 = lambda v: f"{v:.2f}"
M = {"mlNSys": f"{R['n_sistemas']:,}".replace(",", "{,}"), "mlFolds": str(N_PART),
     "mlCorr": f2(R["todas"]["pearson"]), "mlCorrLo": f2(R["todas"]["pearson_ic95"][0]), "mlCorrHi": f2(R["todas"]["pearson_ic95"][1]),
     "mlRTwo": f2(R["todas"]["r2"]), "mlFracCeilPct": f"{100 * R['todas']['fraccion_del_techo']:.0f}",
     "mlImplied": f2(R["todas"]["correlacion_implicita_con_propiedad"]),
     "mlImpliedLo": f2(R["todas"]["correlacion_implicita_ic95"][0]), "mlImpliedHi": f2(min(R["todas"]["correlacion_implicita_ic95"][1], 1)),
     "mlRepN": str(R["replicados"]["n"]), "mlRepA": f2(R["replicados"]["pearson_A"]), "mlRepB": f2(R["replicados"]["pearson_B"]),
     "mlRepMean": f2(R["replicados"]["pearson_media"]), "mlRepMeanExp": f2(R["replicados"]["media_esperada_por_el_modelo_de_medida"]),
     "mlRepGainLo": f2(R["replicados"]["ganancia_media_ic95"][0]), "mlRepGainHi": f2(R["replicados"]["ganancia_media_ic95"][1]),
     "mlLongN": str(R["larga"]["n"])}
for nombre, nn in (("todos", "All"), ("cribado", "Screen"), ("literatura", "Lit")):
    e = R["larga"][nombre]
    M[f"mlLong{nn}Model"] = f2(e["spearman_modelo"]); M[f"mlLong{nn}Short"] = f2(e["spearman_corrida_corta"])
    M[f"mlLong{nn}DiffLo"], M[f"mlLong{nn}DiffHi"] = (f2(v) for v in e["dif_spearman_modelo_menos_corta_ic95"]); M[f"mlLong{nn}N"] = str(e["n"])
F = R["fuga"]
M.update({"mlLeakN": str(F["n_trayectorias_replicadas_con_pareja_en_entrenamiento"]), "mlLeakTraj": f2(F["pearson_division_por_trayectoria"]),
          "mlLeakSys": f2(F["pearson_division_por_sistema"])})
Fm = R["familias"]; M["mlNNSys"] = f2(Fm["vecino_division_por_sistema"])
for nombre, nn in (("finas", "Fine"), ("gruesas", "Coarse")):
    e = Fm[nombre]
    M.update({f"mlFam{nn}Sim": f"{e['similitud_corte']:.1f}", f"mlFam{nn}Groups": f"{e['n_grupos']:,}".replace(",", "{,}"),
              f"mlFam{nn}LargestPct": f"{100 * e['mayor_grupo_frac']:.0f}", f"mlFam{nn}Corr": f2(e["pearson"]),
              f"mlFam{nn}CorrLo": f2(e["pearson_ic95"][0]), f"mlFam{nn}CorrHi": f2(e["pearson_ic95"][1]),
              f"mlFam{nn}FracCeilPct": f"{100 * e['fraccion_del_techo']:.0f}", f"mlFam{nn}NN": f2(e["vecino_en_entrenamiento"]),
              f"mlFam{nn}LongScreen": f2(e["spearman_larga_cribado"])})
with open("paper/numeros_ml.tex", "w") as fh:
    fh.write("% Generado por ml_demo.py a partir de resultados/ml_demo.json. No editar a mano.\n")
    for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

lo1, hi1 = (float(v) for v in H["techo_correlacion_ic95"])                 # ya es una correlación (√r), no una fiabilidad
sb = lambda r: float(np.sqrt(2 * r / (1 + r)))                             # techo de la media de dos corridas (Spearman-Brown)
rlo, rhi = (v ** 2 for v in (lo1, hi1))
techo1 = f"{f2(np.sqrt(r1))} ({f2(lo1)} to {f2(hi1)})"; techo2 = f"{f2(np.sqrt(r2))} ({f2(sb(rlo))} to {f2(sb(rhi))})"
with open("paper/tabla_ml.tex", "w") as fh:
    fh.write("% Generado por ml_demo.py. No editar a mano.\n\\begin{tabular}{lrrr}\n\\toprule\n"
             "Target of the out-of-sample prediction & Systems & Pearson $r$ (95\\% CI) & Ceiling (95\\% CI) \\\\\n\\midrule\n")
    fh.write(f"One 5 ns label, all systems & {M['mlNSys']} & {M['mlCorr']} ({M['mlCorrLo']} to {M['mlCorrHi']}) & {techo1} \\\\\n")
    for nn, txt in (("Fine", "fine"), ("Coarse", "coarse")):
        fh.write(f"\\quad split by {txt} chemical family (Tanimoto $\\geq$ {M[f'mlFam{nn}Sim']}) & {M['mlNSys']} & {M[f'mlFam{nn}Corr']} "
                 f"({M[f'mlFam{nn}CorrLo']} to {M[f'mlFam{nn}CorrHi']}) & {techo1} \\\\\n")
    fh.write(f"One 5 ns label (first / second run), replicated systems & {M['mlRepN']} & {M['mlRepA']} / {M['mlRepB']} & {techo1} \\\\\n")
    lo, hi = R["replicados"]["pearson_media_ic95"]
    fh.write(f"Mean of two 5 ns labels, replicated systems & {M['mlRepN']} & {M['mlRepMean']} ({f2(lo)} to {f2(hi)}) & {techo2} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

print(f"sistemas {R['n_sistemas']} (fallos de SMILES {fallos}); techo √r = {np.sqrt(r1):.2f}")
print(f"1) todas: Pearson {c:.2f} {[round(v, 2) for v in ci]}, R² {R['todas']['r2']:.2f}, {100 * R['todas']['fraccion_del_techo']:.0f} % del techo; "
      f"correlación implícita con la propiedad {R['todas']['correlacion_implicita_con_propiedad']:.2f}")
Q = R["replicados"]
print(f"2) replicados (n={Q['n']}): con A {Q['pearson_A']:.2f}, con B {Q['pearson_B']:.2f}, con la media {Q['pearson_media']:.2f} "
      f"(esperada {Q['media_esperada_por_el_modelo_de_medida']:.2f}); ganancia IC95 {[round(v, 2) for v in Q['ganancia_media_ic95']]}")
for nombre in ("todos", "cribado", "literatura"):
    e = R["larga"][nombre]
    print(f"3) larga {nombre} (n={e['n']}): Spearman modelo {e['spearman_modelo']:.2f} | corrida corta {e['spearman_corrida_corta']:.2f}; "
          f"modelo − corta IC95 {[round(v, 2) for v in e['dif_spearman_modelo_menos_corta_ic95']]}")
for nombre in ("finas", "gruesas"):
    e = R["familias"][nombre]
    print(f"5) familias {nombre} (Tanimoto ≥ {e['similitud_corte']:.1f}): {e['n_grupos']} grupos, mayor {100 * e['mayor_grupo_frac']:.0f} %; "
          f"Pearson {e['pearson']:.2f} {[round(v, 2) for v in e['pearson_ic95']]} ({100 * e['fraccion_del_techo']:.0f} % del techo); "
          f"vecino en entrenamiento {e['vecino_en_entrenamiento']:.2f} (por sistema {R['familias']['vecino_division_por_sistema']:.2f}); "
          f"Spearman larga cribado {e['spearman_larga_cribado']:.2f}")
print(f"4) fuga (n={F['n_trayectorias_replicadas_con_pareja_en_entrenamiento']} trayectorias replicadas con su pareja en entrenamiento): "
      f"división por trayectoria {F['pearson_division_por_trayectoria']:.2f} | por sistema {F['pearson_division_por_sistema']:.2f}; "
      f"todas por trayectoria {F['pearson_todas_division_por_trayectoria']:.2f}")
