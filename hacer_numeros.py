"""Genera paper/numeros.tex (macros de LaTeX) a partir de resultados/preliminar.json, para que ninguna cifra del
manuscrito se escriba a mano. Ejecutar después de analizar_curvas.py."""
import json
R = json.load(open("resultados/preliminar.json")); L = R["largas"]; lag = {t: i for i, t in enumerate(L["retardos_ns"])}
pct = lambda x: f"{100 * x:.1f}"; f2 = lambda x: f"{x:.2f}"; n = lambda x: f"{x:,}".replace(",", "{,}")
M = {"nShort": n(R["n_5ns"]), "nLong": n(R["n_50ns"]), "nTotal": n(R["n_5ns"] + R["n_50ns"]),
     "betaShortMed": f2(R["beta_5ns"]["mediana"]), "betaShortLo": f2(R["beta_5ns"]["p10"]), "betaShortHi": f2(R["beta_5ns"]["p90"]),
     "betaShortSubPct": pct(R["beta_5ns"]["frac_subdifusivo_0.7"]), "betaShortDiffPct": f"{100 * R['beta_5ns']['frac_difusivo_0.9']:.2f}",
     "betaShortDiffN": str(round(R["beta_5ns"]["frac_difusivo_0.9"] * R["n_5ns"])),
     "betaLongMed": f2(R["beta_50ns"]["mediana"]), "betaLongLo": f2(R["beta_50ns"]["p10"]), "betaLongHi": f2(R["beta_50ns"]["p90"]),
     "betaLongSubPct": pct(R["beta_50ns"]["frac_subdifusivo_0.7"]), "betaLongDiffPct": pct(R["beta_50ns"]["frac_difusivo_0.9"]),
     "betaLongDiffN": str(round(R["beta_50ns"]["frac_difusivo_0.9"] * R["n_50ns"])),
     "epSd": f2(R["extremos_5ns"]["sd_log"]), "epFactor": f2(R["extremos_5ns"]["factor_tipico"]),
     "epOutOnePointFivePct": pct(R["extremos_5ns"]["frac_fuera_factor_1.5"]), "epOutTwoPct": pct(R["extremos_5ns"]["frac_fuera_factor_2"]),
     "epRho": f2(R["extremos_5ns"]["spearman_extremos_vs_medio"])}
nombres = {0.5: "Half", 1.0: "One", 2.0: "Two", 5.0: "Five", 10.0: "Ten", 15.0: "Fifteen", 20.0: "Twenty"}
for t, nom in nombres.items():
    j = lag[t]
    M[f"ratio{nom}"] = f2(L["razon_mediana"][j]); M[f"ratio{nom}Lo"] = f2(L["razon_p10"][j]); M[f"ratio{nom}Hi"] = f2(L["razon_p90"][j])
    M[f"rho{nom}"] = f2(L["spearman_vs_25ns"][j]); M[f"within{nom}Pct"] = pct(L["frac_dentro_25pct"][j])
if R.get("pares", {}).get("n", 0) >= 10:
    P = R["pares"]; M["nPairs"] = str(P["n"])
    for col, nom in (("D5_extremos", "PairEnd"), ("D5_medio", "PairAvg")):
        M[f"{nom}Ratio"] = f2(P[col]["razon_mediana"]); M[f"{nom}Lo"] = f2(P[col]["p10"]); M[f"{nom}Hi"] = f2(P[col]["p90"]); M[f"{nom}Rho"] = f2(P[col]["spearman"])
        M[f"{nom}Recall"] = f"{100 * P[col]['recall_top20']:.0f}"
if "conjuntos" in R:
    C = R["conjuntos"]; M["nLongLit"] = str(C["largas_literatura"]); M["nLongScreen"] = str(C["largas_cribado"])
if "replicas" in R:
    Q = R["replicas"]; M["nRep"] = str(Q["n_sistemas"]); M["sdLong"] = f2(Q["sd_log_largas"])
    for col, nom in (("D_extremos", "RepEnd"), ("D_medio", "RepAvg")):
        M[f"{nom}Rho"] = f2(Q[col]["spearman"]); M[f"{nom}Factor"] = f2(Q[col]["factor_tipico"]); M[f"{nom}Sd"] = f2(Q[col]["sd_una_corrida"])
        M[f"{nom}DiffPct"] = f"{100 * Q[col]['frac_difieren_factor_1.5']:.0f}"; M[f"{nom}Icc"] = f2(Q[col]["fiabilidad_icc"])
        M[f"{nom}SdAll"] = f2(Q[col]["sd_log_entre_corridas_todas"]); M[f"{nom}SdSub"] = f2(Q[col]["sd_log_subconjunto"])
    M["RepBetaSd"] = f2(Q["beta_medio"]["sd_una_corrida"]); M["RepBetaRho"] = f2(Q["beta_medio"]["spearman"])
import os
if os.path.exists("resultados/prefijos.json"):
    X = json.load(open("resultados/prefijos.json")); V = X["validacion"]
    M["nPrefix"] = str(X["n"]); M["valMaxDev"] = f"{V['desvio_max_peor']:.0e}".replace("e-0", "e-")
    M["driftMed"] = f"{V['inflado_sin_deriva_mediana']:.1f}"; M["driftMax"] = f"{V['inflado_sin_deriva_max']:.0f}"; M["driftFracPct"] = f"{100 * V['frac_inflado_mayor_10pct']:.0f}"
    fila = lambda k, T: next(x for x in X["estimadores"][k] if x["T"] == T)
    for k, nom in (("extremos", "End"), ("mitad", "Half"), ("ajuste", "Fit"), ("potencial", "Pow")):
        for T, tn in ((2, "Two"), (5, "Five"), (10, "Ten"), (20, "Twenty")):
            M[f"pre{nom}Ratio{tn}"] = f2(fila(k, T)["razon_mediana"]); M[f"pre{nom}Err{tn}"] = f2(fila(k, T)["factor_error"])
            M[f"pre{nom}Cov{tn}"] = f"{100 * fila(k, T)['cobertura_ic95']:.0f}"
    nom_regla = {"regimen_beta_0.9": "BetaNine", "regimen_beta_0.8": "BetaEight", "precision_cv_10": "CvTen", "precision_cv_20": "CvTwenty",
                 "precision_cv_10_potencial": "CvPow"}
    for x in X["reglas"]:
        if x["regla"] not in nom_regla: continue
        nom = nom_regla[x["regla"]]
        M[f"rule{nom}StopPct"] = f"{100 * x['frac_para']:.0f}"; M[f"rule{nom}Time"] = f"{x['T_medio_total']:.1f}"
        M[f"rule{nom}Err"] = f2(x["factor_error"]); M[f"rule{nom}ErrStop"] = f2(x["factor_error_si_para"]) if x["factor_error_si_para"] else "n/a"
        M[f"rule{nom}TimeStop"] = f"{x['T_medio_si_para']:.1f}" if x["T_medio_si_para"] else "n/a"
if os.path.exists("resultados/sensibilidad.json"):                       # generado por sensibilidad.py
    S = json.load(open("resultados/sensibilidad.json"))["ventanas"]
    sub = [v["50"]["sub_0.7"] for v in S]; dif = [v["50"]["dif_0.9"] for v in S]
    M["sensLongSubMinPct"] = f"{100 * min(sub):.0f}"; M["sensLongSubMaxPct"] = f"{100 * max(sub):.0f}"
    M["sensLongDiffMinN"] = str(round(min(dif) * R["n_50ns"])); M["sensLongDiffMaxN"] = str(round(max(dif) * R["n_50ns"]))
if os.path.exists("resultados/incertidumbre.json"):                      # generado por incertidumbre.py
    U = json.load(open("resultados/incertidumbre.json"))
    for col, nom in (("D_extremos", "RepEnd"), ("D_medio", "RepAvg")):
        Q = U["replicas"][col]
        M[f"{nom}IccLo"], M[f"{nom}IccHi"] = (f2(v) for v in Q["fiabilidad_ic95"])
        M[f"{nom}IccSub"] = f2(Q["icc1_subconjunto"]); M[f"{nom}IccSubLo"], M[f"{nom}IccSubHi"] = (f2(v) for v in Q["icc1_subconjunto_ic95"])
        M[f"{nom}PairFactor"] = f2(Q["factor_diferencia_1sd"]); M[f"{nom}PairFactorNinetyFive"] = f2(Q["factor_diferencia_95"])
    for col, nom in (("D5_extremos", "PairEnd"), ("D5_medio", "PairAvg")):
        Q = U["pares"][col]
        M[f"{nom}RhoLo"], M[f"{nom}RhoHi"] = (f2(v) for v in Q["spearman_ic95"])
        M[f"{nom}RhoScreen"] = f2(Q["spearman_cribado"]); M[f"{nom}RhoLit"] = f2(Q["spearman_literatura"])
        for frac, fn in ((5, "Five"), (10, "Ten"), (20, "Twenty")):
            s = Q[f"top{frac}"]; M[f"{nom}Top{fn}K"] = str(s["k"]); M[f"{nom}Top{fn}Hits"] = str(s["aciertos"])
if os.path.exists("resultados/tamanos.csv"):                              # generado por tamanos.py (Miti)
    import pandas as pd
    E = pd.read_csv("resultados/tamanos.csv")
    n0 = lambda x: f"{x:,.0f}".replace(",", "{,}")
    M.update({"tamNAtomsMed": n0(E.n_atomos.median()), "tamNAtomsMin": n0(E.n_atomos.min()), "tamNAtomsMax": n0(E.n_atomos.max()),
              "tamChainsMed": f"{E.n_cadenas.median():.0f}", "tamChainsMin": f"{E.n_cadenas.min():.0f}", "tamChainsMax": f"{E.n_cadenas.max():.0f}",
              "tamBoxMin": f"{E.lx_A.min() / 10:.1f}", "tamBoxMax": f"{E.lx_A.max() / 10:.1f}", "tamBoxMed": f"{E.lx_A.median() / 10:.1f}",
              "tamNIons": f"{E.n_li.median():.0f}", "tamIonsAllEqual": "1" if E.n_li.nunique() == 1 and E.n_tfsi.nunique() == 1 else "0",
              "tamNRecount": str(int((E.fuente_iones_cadenas == "recuento").sum()))})
with open("paper/numeros.tex", "w") as fh:
    fh.write("% Generado por hacer_numeros.py a partir de resultados/preliminar.json. No editar a mano.\n")
    for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros"); print(open("paper/numeros.tex").read()[:900])
