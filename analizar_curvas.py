"""Análisis preliminar con las curvas MSD(τ) de HTP-MD (6.137 simulaciones de 5 ns y 133 de 50 ns, 353 K, PCFF+).

Preguntas:
 1. ¿En qué régimen está el litio al final de cada simulación? (exponente local β = d ln MSD / d ln τ)
 2. En las simulaciones largas, ¿cuánto cambia la difusividad aparente D_app(τ) = MSD(τ)/(6τ) con el tiempo observado?
    ¿A partir de qué τ el ranking de electrolitos se estabiliza?
 3. ¿Cuánto ruido añade estimar D solo con el primer y el último fotograma (como hace HTP-MD)?
 4. Si existe data/rutas.csv: mismas formulaciones a 5 ns y a 50 ns.

Salida: resultados/preliminar.json, resultados/tabla_por_trayectoria.csv y figuras en paper/figures/.
"""
import glob, json, os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, NARANJA, VERDE, TINTA, TINTA2, REJILLA = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TINTA2,
                     "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True,
                     "grid.color": REJILLA, "grid.linewidth": 0.6, "lines.linewidth": 2.0, "figure.dpi": 150,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
os.makedirs("resultados", exist_ok=True); os.makedirs("paper/figures", exist_ok=True)


def exponente(t, m, a, b):
    """Pendiente log-log de MSD entre las fracciones a y b de la duración."""
    k = (t >= a * t[-1]) & (t <= b * t[-1]) & (m > 0)
    return np.polyfit(np.log(t[k]), np.log(m[k]), 1)[0] if k.sum() >= 5 else np.nan


curvas, filas = {}, []
for f in glob.glob("data/curvas/*.json"):
    d = json.load(open(f)); ns = d["condiciones"]["simulation_length"]
    for ion in ("li", "tfsi"):
        t = np.array(d[ion]["time"], float); m = np.array(d[ion]["msd"], float)
        t = t / t[-1] * ns                                             # eje en ns: el último retardo es la duración completa
        curvas[(int(d["id"]), ion)] = (t, m)
    t, m = curvas[(int(d["id"]), "li")]
    cola = exponente(t, m, 0.2, 0.6)                                   # ajuste potencial en la zona bien promediada
    k = (t >= 0.2 * t[-1]) & (t <= 0.6 * t[-1])
    c = np.polyfit(np.log(t[k]), np.log(m[k]), 1)
    filas.append({"id": int(d["id"]), "ns": ns,
                  "beta_medio": cola,                                   # exponente con retardos entre 20 % y 60 % de la duración
                  "beta_final": exponente(t, m, 0.5, 1.0),
                  "D_extremos": m[-1] / (6 * t[-1]),                    # como HTP-MD: desplazamiento primer→último fotograma (Å²/ns)
                  "D_medio": np.interp(0.5 * t[-1], t, m) / (6 * 0.5 * t[-1]),   # MSD promediado en orígenes, retardo = mitad
                  "resid_extremos": np.log(m[-1]) - np.polyval(c, np.log(t[-1])),  # desviación del último punto respecto a la ley potencial
                  "msd_final": m[-1],
                  "D_tfsi_medio": np.interp(0.5 * t[-1], *curvas[(int(d["id"]), "tfsi")]) / (6 * 0.5 * t[-1])})
T = pd.DataFrame(filas).sort_values("id")
c5, c50 = T[T.ns == 5], T[T.ns == 50]
R = {"n_5ns": len(c5), "n_50ns": len(c50)}

# ---------- 1. Régimen ----------
for nombre, g in (("5ns", c5), ("50ns", c50)):
    b = g.beta_medio
    R[f"beta_{nombre}"] = {"mediana": b.median(), "p10": b.quantile(.1), "p90": b.quantile(.9),
                           "frac_difusivo_0.9": float((b >= 0.9).mean()), "frac_subdifusivo_0.7": float((b < 0.7).mean())}
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for g, col, lab in ((c5, AZUL, f"5 ns runs (n = {len(c5):,})"), (c50, NARANJA, f"50 ns runs (n = {len(c50)})")):
    x = np.sort(g.beta_medio.dropna()); ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, label=lab)
ax.axvline(1.0, color=TINTA2, lw=0.8, ls=(0, (3, 3))); ax.text(1.01, 0.04, "diffusive\n(β = 1)", color=TINTA2, fontsize=7.5, va="bottom")
ax.set_xlabel("Effective MSD exponent β of Li$^+$ (lags 20 to 60% of run)"); ax.set_ylabel("Cumulative fraction of systems")
ax.set_xlim(0, 1.3); ax.set_ylim(0, 1); ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, handlelength=1.4, columnspacing=1.2)
fig.savefig("paper/figures/fig1_exponent_ecdf.pdf"); fig.savefig("paper/figures/fig1_exponent_ecdf.png"); plt.close(fig)

# ---------- 2. Simulaciones largas: D aparente frente al tiempo observado ----------
rejilla = np.array([0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25])
Dapp = np.array([[np.interp(tau, *curvas[(i, "li")]) / (6 * tau) for tau in rejilla] for i in c50.id])
ref = Dapp[:, -1]                                                      # referencia: retardo de 25 ns (mitad de la simulación larga)
razon = Dapp / ref[:, None]
rho = [stats.spearmanr(Dapp[:, j], ref)[0] for j in range(len(rejilla))]
R["largas"] = {"retardos_ns": rejilla.tolist(),
               "razon_mediana": np.median(razon, 0).tolist(), "razon_p10": np.quantile(razon, .1, 0).tolist(), "razon_p90": np.quantile(razon, .9, 0).tolist(),
               "spearman_vs_25ns": [float(x) for x in rho],
               "frac_dentro_25pct": [float((np.abs(razon[:, j] - 1) <= 0.25).mean()) for j in range(len(rejilla))]}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
ax = axs[0]
G = R["largas"]                                                        # la figura sale de los valores guardados (se puede redibujar sin data/)
ax.fill_between(rejilla, G["razon_p10"], G["razon_p90"], color=AZUL, alpha=0.18, lw=0)
ax.plot(rejilla, G["razon_mediana"], color=AZUL, marker="o", ms=4)
ax.axhline(1, color=TINTA2, lw=0.8, ls=(0, (3, 3)))
ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 5, 10, 25]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "25"])
ax.set_xlabel("Observation lag τ (ns)"); ax.set_ylabel("$D_{app}(τ)\\,/\\,D_{app}(25\\,ns)$")
ax.text(0.55, G["razon_p90"][0] * 0.97, f"median and 10th to 90th\npercentile band ({len(c50)} systems)", color=TINTA2, fontsize=7.5, va="top")
ax = axs[1]
ax.plot(rejilla, G["spearman_vs_25ns"], color=NARANJA, marker="o", ms=4)
ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 5, 10, 25]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "25"])
ax.set_ylim(min(0.5, min(G["spearman_vs_25ns"]) - 0.05), 1.01); ax.set_xlabel("Observation lag τ (ns)"); ax.set_ylabel("Spearman ρ with ranking at 25 ns")
fig.tight_layout(); fig.savefig("paper/figures/fig2_long_runs.pdf"); fig.savefig("paper/figures/fig2_long_runs.png"); plt.close(fig)

# ---------- 3. Ruido del estimador de extremos ----------
for nombre, g in (("5ns", c5), ("50ns", c50)):
    r = g.resid_extremos.dropna()
    R[f"extremos_{nombre}"] = {"sd_log": float(r.std()), "factor_tipico": float(np.exp(r.std())), "mediana_log": float(r.median()),
                               "frac_fuera_factor_1.5": float((np.abs(r) > np.log(1.5)).mean()), "frac_fuera_factor_2": float((np.abs(r) > np.log(2)).mean()),
                               "spearman_extremos_vs_medio": float(stats.spearmanr(g.D_extremos, g.D_medio)[0])}
fig, ax = plt.subplots(figsize=(3.4, 2.6))
bins = np.linspace(-1.5, 1.5, 61)
ax.hist(c5.resid_extremos.clip(-1.5, 1.5), bins=bins, color=AZUL, alpha=0.85, lw=0)
for fx in (1 / 2, 2):
    ax.axvline(np.log(fx), color=TINTA2, lw=0.8, ls=(0, (3, 3)))
ax.text(np.log(2) + 0.03, ax.get_ylim()[1] * 0.92, "×2", color=TINTA2, fontsize=7.5); ax.text(np.log(.5) - 0.03, ax.get_ylim()[1] * 0.92, "÷2", color=TINTA2, fontsize=7.5, ha="right")
ax.set_xlabel("ln(end-point MSD / power-law fit), 5 ns runs"); ax.set_ylabel("Number of systems")
fig.savefig("paper/figures/fig3_endpoint_noise.pdf"); fig.savefig("paper/figures/fig3_endpoint_noise.png"); plt.close(fig)

# ---------- 4. Mismo sistema a 5 ns y a 50 ns, y réplicas independientes a 5 ns ----------
ru = pd.read_csv("data/rutas.csv").dropna() if os.path.exists("data/rutas.csv") else pd.DataFrame()
if len(ru) >= 0.99 * len(T):                                           # solo con la descarga de rutas completa
    ru["tanda"] = ru.ruta.str.split("/").str[1]; ru["carpeta"] = ru.ruta.str.split("/").str[2]
    ru["sistema"] = ru.ruta.str.split("/").str[-2]                     # identificador del sistema (p. ej. 9-0-1119513647-0 o PSBDC)
    T = T.merge(ru[["id", "tanda", "carpeta", "sistema"]], on="id", how="left")
    T["literatura"] = T.carpeta.str.startswith("literature")
    R["conjuntos"] = {"largas_literatura": int(((T.ns == 50) & T.literatura).sum()), "largas_cribado": int(((T.ns == 50) & ~T.literatura).sum()),
                      "cortas_literatura": int(((T.ns == 5) & T.literatura).sum()), "cortas_cribado": int(((T.ns == 5) & ~T.literatura).sum())}
    cortas = T[T.ns == 5].sort_values("id")
    a = cortas.groupby("sistema").agg(D5_extremos=("D_extremos", "first"), D5_medio=("D_medio", "first"), n5=("id", "size"))   # primera corrida corta
    b = T[T.ns == 50].set_index("sistema")[["D_medio", "D_extremos", "beta_medio"]].rename(columns={"D_medio": "D50_medio", "D_extremos": "D50_extremos", "beta_medio": "beta50"})
    P = a.join(b, how="inner")
    R["pares"] = {"n": len(P)}
    if len(P) >= 10:
        for col in ("D5_extremos", "D5_medio"):
            lr = np.log(P[col] / P.D50_medio)
            R["pares"][col] = {"razon_mediana": float(np.exp(lr.median())), "p10": float(np.exp(lr.quantile(.1))), "p90": float(np.exp(lr.quantile(.9))),
                               "spearman": float(stats.spearmanr(P[col], P.D50_medio)[0]),
                               "sd_log_alrededor_del_sesgo": float((lr - lr.median()).std()),
                               "frac_dentro_factor_2_del_sesgo": float((np.abs(lr - lr.median()) < np.log(2)).mean())}
        k = max(1, round(0.2 * len(P)))                                # ¿la corta acierta el 20 % mejor según la larga?
        mejores = set(P.D50_medio.nlargest(k).index)
        for col in ("D5_extremos", "D5_medio"):
            R["pares"][col]["recall_top20"] = float(len(mejores & set(P[col].nlargest(k).index)) / k)
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        ax.scatter(P.D50_medio, P.D5_extremos, s=14, color=AZUL, edgecolor="#fcfcfb", lw=0.5)
        lim = [min(P.D50_medio.min(), P.D5_extremos.min()) * 0.7, max(P.D50_medio.max(), P.D5_extremos.max()) * 1.4]
        ax.plot(lim, lim, color=TINTA2, lw=0.8, ls=(0, (3, 3))); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("$D_{app}$, 50 ns run at lag 25 ns (Å$^2$/ns)"); ax.set_ylabel("End-point $D_{app}$, 5 ns run (Å$^2$/ns)")
        fig.savefig("paper/figures/fig4_short_vs_long.pdf"); fig.savefig("paper/figures/fig4_short_vs_long.png"); plt.close(fig)
    # réplicas: dos corridas de 5 ns del mismo sistema con configuración inicial distinta
    rep = cortas.groupby("sistema").filter(lambda g: len(g) == 2).sort_values(["sistema", "id"])
    if len(rep) >= 20:
        R["replicas"] = {"n_sistemas": int(rep.sistema.nunique())}
        for col in ("D_extremos", "D_medio", "beta_medio"):
            x = rep.groupby("sistema")[col].first(); y = rep.groupby("sistema")[col].last()
            d = (np.log(x / y) if col != "beta_medio" else (x - y))
            R["replicas"][col] = {"spearman": float(stats.spearmanr(x, y)[0]),
                                  "sd_una_corrida": float(d.std() / np.sqrt(2)),                       # dispersión atribuible a una sola corrida
                                  "factor_tipico": float(np.exp(d.std() / np.sqrt(2))) if col != "beta_medio" else None,
                                  "frac_difieren_factor_1.5": float((np.abs(d) > np.log(1.5)).mean()) if col != "beta_medio" else None}
            if col != "beta_medio":                                    # fiabilidad de una corrida: varianza entre sistemas / varianza total
                ruido = d.std() / np.sqrt(2); total = np.log(cortas[col]).std()
                R["replicas"][col].update({"sd_log_entre_corridas_todas": float(total), "sd_log_subconjunto": float(np.log(rep.groupby("sistema")[col].mean()).std()),
                                           "fiabilidad_icc": float(1 - ruido ** 2 / total ** 2)})
        R["replicas"]["sd_log_largas"] = float(np.log(T[T.ns == 50].D_medio).std())
T.to_csv("resultados/tabla_por_trayectoria.csv", index=False)
json.dump(R, open("resultados/preliminar.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
