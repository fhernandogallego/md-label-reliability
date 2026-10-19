"""Generalidad: ¿se mantienen los resultados principales con otra propiedad, la difusividad del anión TFSI?

Para comparar en igualdad de condiciones se usa para los dos iones el mismo estimador, el promediado en orígenes a T/2
(D_medio para el Li, D_tfsi_medio para el TFSI; el end-point del TFSI no está en los datos distribuidos). Para cada ion:
  - ruido de una corrida (94 sistemas replicados), fiabilidad r con intervalo bootstrap por sistemas y techo √r;
  - dependencia del ruido con la movilidad (mitad lenta frente a rápida);
  - selección: Spearman de la corrida corta con la referencia de 50 ns (cribado y literatura);
  - aprendizaje automático: random forest con huellas Morgan del monómero, validación cruzada dividida por sistema;
    correlación con una etiqueta y con la media de dos réplicas (y la esperada por el modelo de medida);
  - asignación adaptativa a la frontera frente a réplicas uniformes (top 10 %, ruido homocedástico, que en el Li dio
    lo mismo que el heterocedástico): presupuesto que iguala a cuatro réplicas uniformes.
Lee resultados/tabla_por_trayectoria.csv y resultados/estructuras.csv; necesita rdkit (entorno .venv).
Uso: .venv/bin/python generalidad.py → resultados/generalidad.json, paper/tabla_generalidad.tex, paper/numeros_generalidad.tex
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")
N_BOOT, N_MC, N_PART, SEMILLA, TOP, TANDA = 2000, 100, 5, 20261019, 0.10, 0.05
PRESUPUESTOS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
ETIQUETAS = {"Li": "D_medio", "TFSI": "D_tfsi_medio"}
rng = np.random.default_rng(SEMILLA)
T = pd.read_csv("resultados/tabla_por_trayectoria.csv").merge(pd.read_csv("resultados/estructuras.csv")[["id", "mol_smiles"]], on="id", how="left")
cortas = T[T.ns == 5].sort_values("id").copy()
cortas["original"] = ~cortas.carpeta.str.contains("new_config", na=False)
rep = cortas.groupby("sistema").filter(lambda g: len(g) == 2)
L = T[T.ns == 50].set_index("sistema")
ic = lambda b: [float(np.quantile(b, .025)), float(np.quantile(b, .975))]

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
sis = cortas[cortas.original].drop_duplicates("sistema").set_index("sistema")
X = np.stack([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.float32) for s in sis.mol_smiles])


def fuera_de_muestra(y):
    pred = np.empty(len(y))
    for tr, te in KFold(N_PART, shuffle=True, random_state=SEMILLA).split(y):
        m = RandomForestRegressor(n_estimators=400, max_features=0.3, min_samples_leaf=2, n_jobs=-1, random_state=SEMILLA)
        pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
    return pred


def asignacion(r, s_run, rng):
    """Recall del verdadero top 10 % con réplicas uniformes y con asignación a la frontera (posterior normal, ruido constante)."""
    N = len(cortas); k = int(round(TOP * N)); s_sis = np.sqrt(r * s_run ** 2 / (1 - r))
    def post(suma, n):
        prec = 1 / s_sis ** 2 + n / s_run ** 2; return (suma / s_run ** 2) / prec, 1 / np.sqrt(prec)
    def recall(mu, m): return len(np.intersect1d(np.argpartition(-mu, k)[:k], np.argpartition(-m, k)[:k])) / k
    res = {"uniforme": {}, "frontera": {}}
    for b in PRESUPUESTOS:
        u, f = [], []
        for _ in range(N_MC):
            mu = rng.normal(0, s_sis, N)
            if b == int(b):
                u.append(recall(mu, post(sum(mu + rng.standard_normal(N) * s_run for _ in range(int(b))), np.full(N, b))[0]))
            suma = mu + rng.standard_normal(N) * s_run; n = np.ones(N); resto = int(round((b - 1) * N))
            while resto > 0:
                m, s = post(suma, n); c = np.partition(m, N - k)[N - k]; p = stats.norm.cdf((m - c) / s); lote = min(int(TANDA * N), resto)
                sel = np.argpartition(-(p * (1 - p)), lote)[:lote]; suma[sel] += mu[sel] + rng.standard_normal(lote) * s_run; n[sel] += 1; resto -= lote
            f.append(recall(mu, post(suma, n)[0]))
        if u: res["uniforme"][str(b)] = float(np.mean(u))
        res["frontera"][str(b)] = float(np.mean(f))
    Bs = np.array(PRESUPUESTOS); fr = np.maximum.accumulate([res["frontera"][str(b)] for b in PRESUPUESTOS])
    meta = res["uniforme"]["4.0"]; res["presupuesto_iguala_4"] = float(np.interp(meta, fr, Bs)); res["ahorro_frente_a_4"] = 4 / res["presupuesto_iguala_4"]
    return res


R = {"estimador": "origin-averaged at T/2", "n_replicados": int(rep.sistema.nunique())}
for ion, col in ETIQUETAS.items():
    e = {}
    a = np.log(rep[rep.original].set_index("sistema")[col]); b = np.log(rep[~rep.original].set_index("sistema")[col])
    s2 = [s for s in a.index if s in b.index]; xa, xb = a[s2].values, b[s2].values; todas = np.log(cortas[col].values)
    def fiab(x, y, t): sr = (x - y).std(ddof=1) / np.sqrt(2); return sr, 1 - sr ** 2 / t.var(ddof=1)
    sr, r = fiab(xa, xb, todas); bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(xa), len(xa)); j = rng.integers(0, len(todas), len(todas)); bs.append(fiab(xa[i], xb[i], todas[j])[1])
    e.update({"sd_run": float(sr), "fiabilidad": float(r), "fiabilidad_ic95": ic(bs), "techo": float(np.sqrt(r)),
              "fiabilidad_2rep": float(2 * r / (1 + r))})
    nivel = (xa + xb) / 2; mitad = nivel < np.median(nivel)
    e["sd_lentos"] = float((xa - xb)[mitad].std(ddof=1) / np.sqrt(2)); e["sd_rapidos"] = float((xa - xb)[~mitad].std(ddof=1) / np.sqrt(2))
    # selección frente a la referencia larga (corrida corta original del mismo sistema)
    s3 = [s for s in L.index if s in sis.index]; ref = np.log(L.loc[s3, col].values); corta = np.log(sis.loc[s3, col].values)
    lit = L.loc[s3, "literatura"].values.astype(bool)
    for nombre, sel in (("cribado", ~lit), ("literatura", lit)):
        e[f"spearman_larga_{nombre}"] = float(stats.spearmanr(corta[sel], ref[sel])[0])
    # aprendizaje automático
    y = np.log(sis[col].values); pred = pd.Series(fuera_de_muestra(y), index=sis.index)
    e["ml_pearson"] = float(np.corrcoef(pred.values, y)[0, 1])
    pa = pred[s2].values
    e["ml_rep_una"] = float((np.corrcoef(pa, xa)[0, 1] + np.corrcoef(pa, xb)[0, 1]) / 2)
    e["ml_rep_media"] = float(np.corrcoef(pa, (xa + xb) / 2)[0, 1])
    e["ml_rep_media_esperada"] = e["ml_rep_una"] * np.sqrt(e["fiabilidad_2rep"] / r)
    # asignación
    e["asignacion"] = asignacion(r, sr, np.random.default_rng(SEMILLA))
    R[ion] = e
json.dump(R, open("resultados/generalidad.json", "w"), indent=1)

f2 = lambda v: f"{v:.2f}"
filas = [("Run-to-run noise (log)", lambda e: f2(e["sd_run"])),
         ("Noise, slow / fast half", lambda e: f"{f2(e['sd_lentos'])} / {f2(e['sd_rapidos'])}"),
         ("Reliability (95\\% CI)", lambda e: f"{f2(e['fiabilidad'])} ({f2(e['fiabilidad_ic95'][0])} to {f2(e['fiabilidad_ic95'][1])})"),
         ("Noise ceiling", lambda e: f2(e["techo"])),
         ("Random forest vs one label", lambda e: f2(e["ml_pearson"])),
         ("Random forest vs two-run mean (expected)", lambda e: f"{f2(e['ml_rep_media'])} ({f2(e['ml_rep_media_esperada'])})"),
         ("Spearman with 50 ns run, screening", lambda e: f2(e["spearman_larga_cribado"])),
         ("Spearman with 50 ns run, literature", lambda e: f2(e["spearman_larga_literatura"])),
         ("Runs matching 4 uniform replicates (top 10\\%)",
          lambda e: f"{e['asignacion']['presupuesto_iguala_4']:.1f} ($\\times${e['asignacion']['ahorro_frente_a_4']:.1f})")]
with open("paper/tabla_generalidad.tex", "w") as fh:
    fh.write("% Generado por generalidad.py. No editar a mano.\n\\begin{tabular}{lrr}\n\\toprule\n & \\Li{} & TFSI$^-$ \\\\\n\\midrule\n")
    for nombre, f in filas: fh.write(f"{nombre} & {f(R['Li'])} & {f(R['TFSI'])} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
M = {}
for ion, nn in (("Li", "Li"), ("TFSI", "Tfsi")):
    e = R[ion]
    M.update({f"gen{nn}Rel": f2(e["fiabilidad"]), f"gen{nn}RelLo": f2(e["fiabilidad_ic95"][0]), f"gen{nn}RelHi": f2(e["fiabilidad_ic95"][1]),
              f"gen{nn}Ceil": f2(e["techo"]), f"gen{nn}Ml": f2(e["ml_pearson"]), f"gen{nn}MlMean": f2(e["ml_rep_media"]),
              f"gen{nn}MlMeanExp": f2(e["ml_rep_media_esperada"]), f"gen{nn}RhoScreen": f2(e["spearman_larga_cribado"]),
              f"gen{nn}Saving": f"{e['asignacion']['ahorro_frente_a_4']:.1f}", f"gen{nn}SdSlow": f2(e["sd_lentos"]), f"gen{nn}SdFast": f2(e["sd_rapidos"])})
with open("paper/numeros_generalidad.tex", "w") as fh:
    fh.write("% Generado por generalidad.py a partir de resultados/generalidad.json. No editar a mano.\n")
    for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
for ion in ETIQUETAS:
    e = R[ion]
    print(f"{ion}: σ_run {e['sd_run']:.2f} (lentos {e['sd_lentos']:.2f}, rápidos {e['sd_rapidos']:.2f}) | r {e['fiabilidad']:.2f} {[round(v, 2) for v in e['fiabilidad_ic95']]} "
          f"| techo {e['techo']:.2f} | RF {e['ml_pearson']:.2f}; réplicas una {e['ml_rep_una']:.2f} → media {e['ml_rep_media']:.2f} (esperada {e['ml_rep_media_esperada']:.2f}) "
          f"| Spearman larga cribado {e['spearman_larga_cribado']:.2f}, literatura {e['spearman_larga_literatura']:.2f} "
          f"| asignación: {e['asignacion']['presupuesto_iguala_4']:.2f} corridas = 4 uniformes (×{e['asignacion']['ahorro_frente_a_4']:.2f})")
