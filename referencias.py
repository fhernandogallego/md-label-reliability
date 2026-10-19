"""Sensibilidad del análisis por prefijos a la referencia (respuesta a la objeción "la referencia favorece a la ley potencial").

La referencia principal (analizar_prefijos.py) es la difusividad aparente a retardo 25 ns de la simulación completa de 50 ns,
que está sobre la misma curva que extrapola la ley potencial. Aquí se repite la comparación frente a:
  dapp_15, dapp_35, dapp_40   difusividad aparente MSD(τ)/(6τ) de la trayectoria completa a otros retardos
  pendiente_tardia            pendiente de un ajuste lineal del MSD completo entre 20 y 40 ns / 6: la estimación "de manual"
                              de la difusividad a partir de la simulación larga; no es una ley potencial
La ley potencial ajustada en el prefijo se extrapola a cada referencia con su propia definición: a MSD(τ)/(6τ) en el retardo
correspondiente, o a la pendiente media A·[τ2^β − τ1^β]/(τ2 − τ1)/6 en la ventana 20–40 ns.

Uso: python referencias.py   → resultados/referencias.json y paper/tabla_referencias.tex
"""
import glob, json, os
import numpy as np
from scipy import stats

DEST = os.environ.get("HTPMD_DEST", "data/iones")
PREFIJOS = [5, 10, 20]
VENTANA_TARDIA = (20.0, 40.0)
REFS = ["dapp_15", "dapp_25", "dapp_35", "dapp_40", "pendiente_tardia"]
N_BOOT, SEMILLA = 2000, 20261016


def msd_por_ion(r):
    """Copia literal de analizar_prefijos.msd_por_ion (ese módulo ejecuta el análisis completo al importarse)."""
    x = np.transpose(r, (1, 0, 2)).astype(np.float64); n = x.shape[1]
    F = np.fft.rfft(x, n=2 * n, axis=1); s2 = np.fft.irfft(F * F.conj(), axis=1)[:, :n, :].sum(-1)
    d = np.square(x).sum(-1); c = np.cumsum(d, axis=1)
    m = np.arange(n)
    primero = c[:, n - 1 - m]
    segundo = c[:, -1][:, None] - np.concatenate([np.zeros((len(c), 1)), c[:, :-1]], axis=1)[:, m]
    return (primero + segundo - 2 * s2) / (n - m)


def ref_y_prediccion(f):
    d = np.load(f); dt = float(d["dt_ps"]) / 1000.0
    li = d["li"].astype(np.float64) - d["com"][:, None, :]
    if (li.shape[0] - 1) * dt < 49: return None
    completo = msd_por_ion(li).mean(0); t = np.arange(li.shape[0]) * dt
    refs = {f"dapp_{int(tau)}": float(np.interp(tau, t, completo) / (6 * tau)) for tau in (15, 25, 35, 40)}
    k = (t >= VENTANA_TARDIA[0]) & (t <= VENTANA_TARDIA[1])
    refs["pendiente_tardia"] = float(np.polyfit(t[k], completo[k], 1)[0] / 6)
    pred = {}
    for T in PREFIJOS:
        nf = int(round(T / dt)) + 1; r = li[:nf]
        curva = msd_por_ion(r).mean(0); tt = np.arange(nf) * dt
        sel = np.unique(np.round(np.geomspace(1, nf - 1, 80)).astype(int)); ts, c = tt[sel], curva[sel]
        e = {"extremos": float(np.square(r[-1] - r[0]).sum(-1).mean() / (6 * T)),
             "mitad": float(np.interp(T / 2, ts, c) / (3 * T))}
        kk = (ts >= 0.2 * T) & (ts <= 0.5 * T); e["ajuste"] = float(np.polyfit(ts[kk], c[kk], 1)[0] / 6)
        kk = (ts >= 0.1 * T) & (ts <= 0.5 * T) & (c > 0); b, lnA = np.polyfit(np.log(ts[kk]), np.log(c[kk]), 1); A = np.exp(lnA)
        # la extrapolación se hace a la misma magnitud que define cada referencia
        pot = {f"dapp_{int(tau)}": float(A * tau ** (b - 1) / 6) for tau in (15, 25, 35, 40)}
        t1, t2 = VENTANA_TARDIA; pot["pendiente_tardia"] = float(A * (t2 ** b - t1 ** b) / (t2 - t1) / 6)
        e["potencial"] = pot; e["beta"] = float(b)
        pred[str(T)] = e
    return {"refs": refs, "pred": pred}


def metricas(x, ref, rng):
    lr = np.log(np.asarray(x) / np.asarray(ref)); err = np.abs(lr)
    boot = [np.exp(np.median(err[i])) for i in rng.integers(0, len(err), (N_BOOT, len(err)))]
    return {"razon_mediana": float(np.exp(np.median(lr))), "p10": float(np.exp(np.quantile(lr, .1))), "p90": float(np.exp(np.quantile(lr, .9))),
            "factor_error": float(np.exp(np.median(err))), "factor_error_ic95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "spearman": float(stats.spearmanr(x, ref)[0])}


if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)
    datos = {}
    if not glob.glob(f"{DEST}/*.npz"):                                    # sin data/: regenerar tabla y macros desde el JSON
        R = json.load(open("resultados/referencias.json")); ids = list(R["por_trayectoria"])
    for f in sorted(glob.glob(f"{DEST}/*.npz")):
        if "parcial" in f: continue
        r = ref_y_prediccion(f)
        if r: datos[os.path.basename(f)[:-4]] = r
    if datos:
        ids = sorted(datos); rng = np.random.default_rng(SEMILLA)
        R = {"n": len(ids), "prefijos_ns": PREFIJOS, "referencias": REFS, "ventana_tardia_ns": VENTANA_TARDIA, "tabla": {}}
        # ¿cuánto se parecen entre sí las referencias? (si la pendiente tardía es mucho menor, el sistema sigue lejos de difundir)
        R["pendiente_tardia_vs_dapp25"] = metricas([datos[i]["refs"]["pendiente_tardia"] for i in ids], [datos[i]["refs"]["dapp_25"] for i in ids], rng)
        for ref in REFS:
            y = [datos[i]["refs"][ref] for i in ids]
            for T in PREFIJOS:
                for est in ("extremos", "mitad", "ajuste", "potencial"):
                    x = [datos[i]["pred"][str(T)][est][ref] if est == "potencial" else datos[i]["pred"][str(T)][est] for i in ids]
                    R["tabla"][f"{ref}|{T}|{est}"] = metricas(x, y, rng)
        # ¿Acotan los dos estimadores baratos la difusividad tardía? extremos (sobreestima) y ley potencial (subestima a tiempos largos)
        for T in PREFIJOS:
            dentro = [datos[i]["pred"][str(T)]["potencial"]["pendiente_tardia"] <= datos[i]["refs"]["pendiente_tardia"] <= datos[i]["pred"][str(T)]["extremos"] for i in ids]
            R[f"acotada_pendiente_{T}ns"] = float(np.mean(dentro))
            dentro = [datos[i]["pred"][str(T)]["potencial"]["dapp_40"] <= datos[i]["refs"]["dapp_40"] <= datos[i]["pred"][str(T)]["extremos"] for i in ids]
            R[f"acotada_dapp40_{T}ns"] = float(np.mean(dentro))
        R["por_trayectoria"] = datos
        json.dump(R, open("resultados/referencias.json", "w"), indent=1)
    # macros para el manuscrito (paper/numeros_referencias.tex), como hacer_numeros.py
    t = R["tabla"]; f2 = lambda x: f"{x:.2f}"; pct = lambda x: f"{100 * x:.0f}"
    otras = ["dapp_15", "dapp_25", "dapp_35", "dapp_40"]
    def rango(est, campo): v = [t[f"{r}|5|{est}"][campo] for r in otras]; return f2(min(v)), f2(max(v))
    M = {"refN": str(R["n"]), "refSlopeVsDapp": f2(R["pendiente_tardia_vs_dapp25"]["razon_mediana"])}
    for est, nom in (("potencial", "Pow"), ("extremos", "End"), ("ajuste", "Fit")):
        M[f"ref{nom}ErrMin"], M[f"ref{nom}ErrMax"] = rango(est, "factor_error")
        M[f"ref{nom}BiasMin"], M[f"ref{nom}BiasMax"] = rango(est, "razon_mediana")
    for T, nom in ((5, "Five"), (10, "Ten"), (20, "Twenty")):
        c = t[f"pendiente_tardia|{T}|potencial"]
        M[f"refPowSlopeErr{nom}"] = f2(c["factor_error"]); M[f"refPowSlopeRatio{nom}"] = f2(c["razon_mediana"])
        M[f"refBracket{nom}Pct"] = pct(R[f"acotada_pendiente_{T}ns"])
    M["refEndSlopeErrFive"] = f2(t["pendiente_tardia|5|extremos"]["factor_error"]); M["refFitSlopeErrFive"] = f2(t["pendiente_tardia|5|ajuste"]["factor_error"])
    with open("paper/numeros_referencias.tex", "w") as fh:
        fh.write("% Generado por referencias.py a partir de resultados/referencias.json. No editar a mano.\n")
        for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("acotada por [ley potencial, extremos]:", {k: round(v, 2) for k, v in R.items() if k.startswith("acotada")})
    NOM = {"dapp_15": "$D_{app}$ at 15 ns", "dapp_25": "$D_{app}$ at 25 ns (main)", "dapp_35": "$D_{app}$ at 35 ns",
           "dapp_40": "$D_{app}$ at 40 ns", "pendiente_tardia": "Slope of MSD, 20 to 40 ns"}
    with open("paper/tabla_referencias.tex", "w") as fh:
        fh.write("% Generado por referencias.py. No editar a mano.\n\\begin{tabular}{lrrrr}\n\\toprule\n"
                 "Reference (from the 50 ns run) & End-point, 5 ns & Linear fit, 5 ns & Power law, 5 ns & Power law, 10 ns \\\\\n\\midrule\n")
        for ref in REFS:
            celdas = [R["tabla"][f"{ref}|{T}|{e}"] for T, e in ((5, "extremos"), (5, "ajuste"), (5, "potencial"), (10, "potencial"))]
            fh.write(NOM[ref] + " & " + " & ".join(f"$\\times${c['factor_error']:.2f} ({c['razon_mediana']:.2f})" for c in celdas) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"{len(ids)} trayectorias | pendiente tardía / D_app(25 ns): mediana {R['pendiente_tardia_vs_dapp25']['razon_mediana']:.2f}")
    for ref in REFS:
        print(f"{ref:17s}", "  ".join(f"{e}@{T}: ×{R['tabla'][f'{ref}|{T}|{e}']['factor_error']:.2f} (sesgo {R['tabla'][f'{ref}|{T}|{e}']['razon_mediana']:.2f}, ρ {R['tabla'][f'{ref}|{T}|{e}']['spearman']:.2f})"
                                      for T, e in ((5, "extremos"), (5, "ajuste"), (5, "potencial"), (10, "potencial"), (20, "potencial"))))
