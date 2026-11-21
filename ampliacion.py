"""Ampliación del piloto SCAYLE: 393 K y caja 2×2×2 (y el protocolo de equilibrado de HTP-MD, que no entra en el artículo).

Lee resultados/etiquetas_ampliacion.csv (ln D en Å²/ns por corrida, end-point y promediado en orígenes a T/2, con los
controles de HTP-MD), extraídas de las trayectorias por simulaciones/analizar_ampliacion.py en el repositorio de trabajo.
Las referencias a 353 K y caja original son, para cada sistema, las etiquetas con la MISMA estructura de partida (original
de HTP-MD, reproducir, vel1, vel2), cuya media tiene poco ruido de dinámica.
  1. htpeq (3 sistemas × 2): estructuras generadas con el protocolo de HTP-MD frente a la original.
  2. T393 (10 × 2): factor de aumento de D a 393 K; ruido de dinámica a 393 K frente al de 353 K; orden de los sistemas.
  3. caja2 (4 × 2): efecto del tamaño en D y ruido de dinámica con 8 veces más iones (≈ 1/√8 si fuera solo estadístico).
Muestras pequeñas: intervalos bootstrap por sistemas, solo descriptivos.
Uso: python ampliacion.py → resultados/ampliacion.json, paper/numeros_ampliacion.tex
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from comun import SEMILLA_BOOT, ic

MISMA_ESTRUCTURA = ["htpmd_original", "reproducir", "vel1", "vel2"]
A = pd.read_csv("resultados/etiquetas_ampliacion.csv")
rng = np.random.default_rng(SEMILLA_BOOT)


def con_ic(P, f, n_boot=4000):
    P = P.dropna(); v = float(f(P))
    b = [f(P.iloc[rng.integers(0, len(P), len(P))]) for _ in range(n_boot)]
    return {"valor": v, "ic95": ic(np.array([x for x in b if np.isfinite(x)])), "n_sistemas": len(P)}


def sd_par(Q, a, b): return np.sqrt(((Q[a] - Q[b]) ** 2).mean() / 2)
def sd_dentro(Q, cols): return np.sqrt(Q[cols].var(1, ddof=1).mean())


R = {}
for est in ("extremos", "medio"):
    P = A.pivot_table(index="sistema", columns="modo", values=est)
    ref = P[[m for m in MISMA_ESTRUCTURA if m in P]].mean(1)
    e = {}
    if {"htpeq1", "htpeq2"} <= set(P):                                           # 1. protocolo de HTP-MD
        Q = P.assign(ref=ref, nuevo=P[["htpeq1", "htpeq2"]].mean(1))[["ref", "nuevo", "htpeq1", "htpeq2", "htpmd_original", "htpmd_replica"]]
        e["htpeq"] = {"razon_frente_a_misma_estructura": con_ic(Q, lambda q: np.exp((q.nuevo - q.ref).mean())),
                      "sd_entre_estructuras_nuevas": con_ic(Q, lambda q: sd_par(q, "htpeq1", "htpeq2")),
                      "sd_replicas_htpmd_mismos_sistemas": con_ic(Q, lambda q: sd_par(q, "htpmd_original", "htpmd_replica"))}
        if {"rec1", "rec2"} <= set(P):
            Qr = P.loc[Q.dropna().index].assign(ref=ref)
            e["htpeq"]["razon_recocido_mismos_sistemas"] = con_ic(Qr, lambda q: np.exp((q[["rec1", "rec2"]].mean(1) - q.ref).mean()))
    t = [m for m in P if m.startswith("T393_")]
    if len(t) == 2:                                                               # 2. 393 K
        Q = P.assign(ref=ref, t393=P[t].mean(1))
        Q = Q[["ref", "t393", *t, *[m for m in MISMA_ESTRUCTURA if m in P]]]
        e["T393"] = {"razon_D_393_353": con_ic(Q, lambda q: np.exp((q.t393 - q.ref).mean())),
                     "sd_dinamica_393": con_ic(Q, lambda q: sd_par(q, t[0], t[1])),
                     "sd_dinamica_353": con_ic(Q, lambda q: sd_dentro(q, [m for m in MISMA_ESTRUCTURA if m in q])),
                     "spearman_orden_353_393": float(stats.spearmanr(Q.dropna().ref, Q.dropna().t393)[0]),
                     "energia_activacion_kJmol_aparente": float(8.314e-3 * np.mean(Q.dropna().t393 - Q.dropna().ref) / (1 / 353 - 1 / 393))}
    k = [m for m in P if m.startswith("caja2_")]
    if len(k) == 2:                                                               # 3. caja 2×2×2
        Q = P.assign(ref=ref, grande=P[k].mean(1))[["ref", "grande", *k, *[m for m in MISMA_ESTRUCTURA if m in P]]]
        e["caja2"] = {"razon_D_grande_original": con_ic(Q, lambda q: np.exp((q.grande - q.ref).mean())),
                      "sd_dinamica_caja_grande": con_ic(Q, lambda q: sd_par(q, k[0], k[1])),
                      "sd_dinamica_caja_original": con_ic(Q, lambda q: sd_dentro(q, [m for m in MISMA_ESTRUCTURA if m in q])),
                      "esperado_si_solo_estadistico": "×1/√8 ≈ 0,35 del ruido en caja original"}
    R[est] = e
json.dump(R, open("resultados/ampliacion.json", "w"), indent=1, ensure_ascii=False)

fmt = lambda d: f"{d['valor']:.2f} ({d['ic95'][0]:.2f} a {d['ic95'][1]:.2f}; n={d['n_sistemas']})"
for est, e in R.items():
    print(f"\n=== {est}")
    for bloque, v in e.items():
        print(f"-- {bloque}")
        for k_, d in v.items(): print(f"   {k_}: {fmt(d) if isinstance(d, dict) else d}")

# ---------- Macros para la información suplementaria (393 K y caja 2×2×2; el protocolo de HTP-MD, n = 3, no se usa) ----------
f2 = lambda v: f"{v:.2f}"
M = {}
for est, nom in (("extremos", "End"), ("medio", "Avg")):
    t, k = R[est]["T393"], R[est]["caja2"]
    M[f"amp{nom}TRatio"] = f"{t['razon_D_393_353']['valor']:.1f}"
    M[f"amp{nom}TRatioLo"], M[f"amp{nom}TRatioHi"] = (f"{v:.1f}" for v in t["razon_D_393_353"]["ic95"])
    M[f"amp{nom}TEa"] = f"{t['energia_activacion_kJmol_aparente']:.0f}"
    M[f"amp{nom}TSdHot"] = f2(t["sd_dinamica_393"]["valor"]); M[f"amp{nom}TSdHotLo"], M[f"amp{nom}TSdHotHi"] = (f2(v) for v in t["sd_dinamica_393"]["ic95"])
    M[f"amp{nom}TSdRef"] = f2(t["sd_dinamica_353"]["valor"])
    M[f"amp{nom}TRho"] = f2(t["spearman_orden_353_393"])
    M[f"amp{nom}BoxRatio"] = f2(k["razon_D_grande_original"]["valor"])
    M[f"amp{nom}BoxRatioLo"], M[f"amp{nom}BoxRatioHi"] = (f2(v) for v in k["razon_D_grande_original"]["ic95"])
    M[f"amp{nom}BoxSdBig"] = f2(k["sd_dinamica_caja_grande"]["valor"]); M[f"amp{nom}BoxSdRef"] = f2(k["sd_dinamica_caja_original"]["valor"])
    M[f"amp{nom}BoxSdBigLo"], M[f"amp{nom}BoxSdBigHi"] = (f2(v) for v in k["sd_dinamica_caja_grande"]["ic95"])
    M[f"amp{nom}BoxEquivRuns"] = f"{(k['sd_dinamica_caja_original']['valor'] / k['sd_dinamica_caja_grande']['valor']) ** 2:.1f}"
M["ampTN"] = str(R["extremos"]["T393"]["razon_D_393_353"]["n_sistemas"]); M["ampBoxN"] = str(R["extremos"]["caja2"]["razon_D_grande_original"]["n_sistemas"])
M["ampStatNoise"] = f2(1 / np.sqrt(8) * R["extremos"]["caja2"]["sd_dinamica_caja_original"]["valor"])
with open("paper/numeros_ampliacion.tex", "w") as fh:
    fh.write("% Generado por ampliacion.py a partir de resultados/etiquetas_ampliacion.csv. No editar a mano.\n")
    for k_, v in M.items(): fh.write(f"\\newcommand{{\\{k_}}}{{{v}}}\n")
