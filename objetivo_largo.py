"""¿Sobrevive el ahorro de la asignación adaptativa si el objetivo es la corrida larga? ¿Qué variabilidad muestra una minicampaña simulada?

R1. Objetivo a tiempo largo. asignacion.py busca el top-k según la etiqueta esperada de 5 ns (μ). Aquí el objetivo es el
    top-k según la referencia de la corrida de 50 ns (L, D_app a 25 ns). Calibración: en los 92 sistemas de cribado con
    réplica y corrida larga, la réplica B parte de una estructura nueva, independiente de la de L; con B = μ + ε,
    corr(B, L) = corr(μ, L)·√r, así que ρ = corr(μ, L) = corr(B, L)/√corr(A, B), con r estimada por la correlación entre
    las dos corridas cortas en el mismo subconjunto. Intervalo por bootstrap de sistemas. En las campañas simuladas,
    L = ρ·μ/σ_sis + √(1−ρ²)·z (solo importa el orden) y se mide el recall del top-k de L con réplicas uniformes y con la
    asignación en la frontera, para ρ estimada, su extremo inferior y ρ = 1 (el caso de asignacion.py).
R2. Detectabilidad aproximada en minicampañas simuladas (validacion_asignacion.py): 92 sistemas, k = 9 (top 10 %) o 18 (top 20 %), 46
    segundas corridas en la frontera o al azar. Se simulan muchas campañas de ese tamaño con el ruido calibrado en los
    pares reales y se mide la ventaja esperada de la frontera sobre el azar (media de 50 asignaciones aleatorias, como en la
    comprobación real), su dispersión entre campañas, la fracción sobre el umbral normal 1,96·SD y una extrapolación orientativa de tamaño.
    No reproduce la prueba bootstrap real ni su calibración bajo la hipótesis nula.

Lee resultados/tabla_por_trayectoria.csv, resultados/techo.json, resultados/incertidumbre.json y
resultados/asignacion.json (para comprobar que la calibración coincide). No necesita data/.
Uso: python objetivo_largo.py → resultados/objetivo_largo.json, paper/numeros_objetivo_largo.tex
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from comun import N_BOOT, SEMILLA_BOOT, ic, originales_cribado

N_MC, N_MC_DET, N_AZAR, TANDA, SEMILLA = 100, 2000, 50, 0.05, 20261118
PRESUPUESTOS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
TOPS = [0.01, 0.10]

T = pd.read_csv("resultados/tabla_por_trayectoria.csv")
H = json.load(open("resultados/techo.json")); U = json.load(open("resultados/incertidumbre.json"))["replicas"]
poblacion = originales_cribado(T); N = len(poblacion)
R = {"N": N, "n_mc": N_MC, "n_mc_detectabilidad": N_MC_DET, "semilla": SEMILLA}

# ---------- R1a. Correlación entre la etiqueta corta esperada y la referencia larga ----------
cortas = T[(T.ns == 5) & ~T.literatura]
A = cortas[~cortas.carpeta.str.contains("new_config", na=False)].set_index("sistema").D_extremos
B = cortas[cortas.carpeta.str.contains("new_config", na=False)].set_index("sistema").D_extremos
L = T[(T.ns == 50) & ~T.literatura].set_index("sistema").D_medio
sis = sorted(set(A.index) & set(B.index) & set(L.index))
a, b, l = (np.log(s[sis].to_numpy()) for s in (A, B, L))


def rho(a, b, l):
    return float(np.corrcoef(b, l)[0, 1] / np.sqrt(np.corrcoef(a, b)[0, 1]))


rng = np.random.default_rng(SEMILLA_BOOT)
boot = [rho(a[i], b[i], l[i]) for i in (rng.integers(0, len(sis), len(sis)) for _ in range(N_BOOT))]
R["rho"] = {"n": len(sis), "valor": rho(a, b, l), "ic95": ic(boot), "corr_B_L": float(np.corrcoef(b, l)[0, 1]),
            "corr_A_B": float(np.corrcoef(a, b)[0, 1])}
ESCENARIOS = {"estimada": min(R["rho"]["valor"], 1.0), "inferior": min(R["rho"]["ic95"][0], 1.0), "uno": 1.0}

# ---------- Modelo de ruido y políticas (las mismas de asignacion.py, etiqueta end-point) ----------
p = H["D_extremos"]; r_, sr = U["D_extremos"]["fiabilidad"], U["D_extremos"]["sd_run"]
lento, rapido = p["heterocedasticidad"]["sd_run_lentos"], p["heterocedasticidad"]["sd_run_rapidos"]
M = {"s_sis": float(np.sqrt(r_ * sr ** 2 / (1 - r_))), "a": (lento + rapido) / 2, "b": (rapido - lento) / (2 * 0.798),
     "sd_run": sr, "fiabilidad": r_}
guardado = json.load(open("resultados/asignacion.json"))["modelo_D_extremos"]
assert all(np.isclose(guardado[k], v) for k, v in M.items()), "calibración distinta de asignacion.py"


def sigma(m): return np.clip(M["a"] + M["b"] * m / M["s_sis"], 0.08, None)


def posterior(suma, n):
    media = suma / np.maximum(n, 1); s = sigma(media)
    prec = 1 / M["s_sis"] ** 2 + n / s ** 2
    return (n * media / s ** 2) / prec, 1 / np.sqrt(prec)


def correr(mu, idx, g): return mu[idx] + g.standard_normal(len(idx)) * sigma(mu[idx])
def mejores(m, k): return np.argpartition(-m, k)[:k]


def uniforme(mu, k, b, g):
    suma = sum(correr(mu, np.arange(N), g) for _ in range(int(b)))
    return mejores(posterior(suma, np.full(N, int(b)))[0], k)


def frontera(mu, k, b, g):
    n = np.ones(N); suma = correr(mu, np.arange(N), g); resto = int(round((b - 1) * N))
    while resto > 0:
        m, s = posterior(suma, n); c = np.partition(m, N - k)[N - k]
        q = stats.norm.cdf((m - c) / s); lote = min(int(TANDA * N), resto)
        sel = np.argpartition(-(q * (1 - q)), lote)[:lote]
        suma[sel] += correr(mu, sel, g); n[sel] += 1; resto -= lote
    return mejores(posterior(suma, n)[0], k)


def igualar(recalls, Bs, meta):
    fr = np.maximum.accumulate(recalls)
    return float(np.interp(meta, fr, Bs)) if meta <= fr[-1] else None


# ---------- R1b. Campañas con objetivo a tiempo largo ----------
R["campanas"] = {}
for nombre, rh in ESCENARIOS.items():
    for top in TOPS:
        k = int(round(top * N)); res = {"uniforme": {}, "frontera": {}}; techo = []
        for j in range(N_MC):
            g0 = np.random.default_rng([SEMILLA, j])                        # mismos sistemas para todas las políticas
            mu = g0.normal(0, M["s_sis"], N); larga = rh * mu / M["s_sis"] + np.sqrt(1 - rh ** 2) * g0.standard_normal(N)
            verdad = mejores(larga, k); techo.append(len(np.intersect1d(verdad, mejores(mu, k))) / k)
            for pol, f, Bs_pol in (("uniforme", uniforme, [1.0, 2.0, 3.0, 4.0]), ("frontera", frontera, PRESUPUESTOS[1:])):
                for bb in Bs_pol:
                    g = np.random.default_rng([SEMILLA, j, int(bb * 100), int(pol == "frontera")])
                    res[pol].setdefault(bb, []).append(len(np.intersect1d(verdad, f(mu, k, bb, g))) / k)
        e = {"rho": rh, "techo_recall": float(np.mean(techo)),
             "uniforme": {str(x): float(np.mean(v)) for x, v in res["uniforme"].items()},
             "frontera": {str(x): float(np.mean(v)) for x, v in res["frontera"].items()}}
        e["frontera"]["1.0"] = e["uniforme"]["1.0"]
        Bs = sorted(float(x) for x in e["frontera"]); meta = e["uniforme"]["4.0"]
        beq = igualar([e["frontera"][str(x)] for x in Bs], Bs, meta)
        e.update({"presupuesto_igual_4": beq, "ahorro_frente_a_4": 4 / beq if beq else None})
        R["campanas"][f"{nombre}|{top}"] = e
        print(f"ρ={rh:.2f} top {top:.0%}: techo {e['techo_recall']:.2f}; uniforme 1/4: {e['uniforme']['1.0']:.2f}/{meta:.2f}; "
              f"frontera iguala 4 con {beq} (ahorro {e['ahorro_frente_a_4']})", flush=True)

# ---------- R2. Detectabilidad aproximada en minicampañas simuladas ----------
ref_par = lambda x, y: (lambda d, nivel: {"s_run": d.std(ddof=1) / np.sqrt(2),
                                          "lento": d[nivel < np.median(nivel)].std(ddof=1) / np.sqrt(2),
                                          "rapido": d[nivel >= np.median(nivel)].std(ddof=1) / np.sqrt(2)})(x - y, (x + y) / 2)
pr = ref_par(a, b); s_tot = float(np.log(poblacion.D_extremos).std(ddof=1))
P = {"s_sis": float(np.sqrt(s_tot ** 2 - pr["s_run"] ** 2)), "a": (pr["lento"] + pr["rapido"]) / 2,
     "b": (pr["rapido"] - pr["lento"]) / (2 * 0.798)}
n_c, extra = len(sis), len(sis) // 2
sg = lambda m: np.clip(P["a"] + P["b"] * m / P["s_sis"], 0.08, None)


def post_c(suma, n):
    media = suma / n; s = sg(media); prec = 1 / P["s_sis"] ** 2 + n / s ** 2
    return (n * media / s ** 2) / prec, 1 / np.sqrt(prec)


def sel_c(y1, y2, usa, k):
    return set(np.argsort(-post_c(y1 + np.where(usa, y2, 0), 1 + usa.astype(float))[0])[:k])


def frontera_c(y1, y2, k, lote=5):
    usa = np.zeros(n_c, bool)
    while usa.sum() < extra:
        m, s = post_c(y1 + np.where(usa, y2, 0), 1 + usa.astype(float)); c = np.sort(m)[-k]
        q = stats.norm.cdf((m - c) / s); duda = np.where(usa, -1, q * (1 - q))
        usa[np.argsort(-duda)[:min(lote, extra - usa.sum())]] = True
    return usa


R["detectabilidad"] = {
    "n_sistemas": n_c, "segundas": extra, "ruido": P,
    "metodo": {"umbral_sd": 1.96, "distribucion_nula_simulada": False,
               "prueba_bootstrap_real_reproducida": False, "ruido_reajustado_por_campana": False,
               "extrapolacion_N": "Gaussian approximation with SD proportional to N^(-1/2); not validated across sample sizes."}
}
g = np.random.default_rng(SEMILLA + 1)
for top in (0.10, 0.20):
    k = max(1, round(top * n_c)); difs = []
    for _ in range(N_MC_DET):
        mu = g.normal(0, P["s_sis"], n_c); y1 = mu + g.standard_normal(n_c) * sg(mu); y2 = mu + g.standard_normal(n_c) * sg(mu)
        verdad = set(np.argsort(-mu)[:k])
        f_ = len(verdad & sel_c(y1, y2, frontera_c(y1, y2, k), k)) / k
        az = np.mean([len(verdad & sel_c(y1, y2, np.isin(np.arange(n_c), g.choice(n_c, extra, replace=False)), k)) / k
                      for _ in range(N_AZAR)])
        difs.append(f_ - az)
    d = np.array(difs); media, sd = float(d.mean()), float(d.std(ddof=1))
    fraccion_umbral = float(np.mean(d > 1.96 * sd))
    n80 = float(n_c * ((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * sd / media) ** 2) if media > 0 else None
    R["detectabilidad"][str(top)] = {"k": k, "ventaja_media": media, "sd_entre_campanas": sd, "fraccion_sobre_umbral_196sd": fraccion_umbral,
                               "prob_ventaja_no_positiva": float(np.mean(d <= 0)), "n80_aproximado": n80}
    print(f"detectabilidad aproximada top {top:.0%} (k={k}): ventaja media {100 * media:.1f} puntos, sd {100 * sd:.1f}, fracción sobre 1.96·SD {fraccion_umbral:.2f}, "
          f"P(≤0) {np.mean(d <= 0):.2f}, escala N80 aproximada: {n80:.0f}", flush=True)

json.dump(R, open("resultados/objetivo_largo.json", "w"), indent=1)

f2 = lambda v: f"{v:.2f}"; pc = lambda v: f"{100 * v:.0f}"; f1 = lambda v: f"{v:.1f}"
Mc = {"olN": str(R["rho"]["n"]), "olRho": f2(R["rho"]["valor"]), "olRhoLo": f2(R["rho"]["ic95"][0]),
      "olRhoHi": f2(min(R["rho"]["ic95"][1], 1.0)), "olCorrBL": f2(R["rho"]["corr_B_L"]), "olCorrAB": f2(R["rho"]["corr_A_B"])}
for nombre, nn in (("estimada", "Est"), ("inferior", "Low"), ("uno", "One")):
    for top, nt in ((0.01, "One"), (0.10, "Ten")):
        e = R["campanas"][f"{nombre}|{top}"]
        Mc.update({f"ol{nn}Top{nt}Ceil": pc(e["techo_recall"]), f"ol{nn}Top{nt}UnifOne": pc(e["uniforme"]["1.0"]),
                   f"ol{nn}Top{nt}UnifFour": pc(e["uniforme"]["4.0"]), f"ol{nn}Top{nt}BoundTwo": pc(e["frontera"]["2.0"]),
                   f"ol{nn}Top{nt}Eq": f1(e["presupuesto_igual_4"]) if e["presupuesto_igual_4"] else "n/a",
                   f"ol{nn}Top{nt}Saving": f1(e["ahorro_frente_a_4"]) if e["ahorro_frente_a_4"] else "n/a"})
for top, nt in ((0.10, "Ten"), (0.20, "Twenty")):
    e = R["detectabilidad"][str(top)]
    Mc.update({f"olDet{nt}Adv": f1(100 * e["ventaja_media"]), f"olDet{nt}Sd": f1(100 * e["sd_entre_campanas"]),
               f"olDet{nt}Pct": pc(e["fraccion_sobre_umbral_196sd"]), f"olDet{nt}NoGainPct": pc(e["prob_ventaja_no_positiva"]),
               f"olDet{nt}NScale": f"{100 * round(e['n80_aproximado'] / 100):,}".replace(",", "{,}") if e["n80_aproximado"] else "n/a"})
with open("paper/numeros_objetivo_largo.tex", "w") as fh:
    fh.write("% Generado por objetivo_largo.py a partir de resultados/objetivo_largo.json. No editar a mano.\n")
    for kk, v in Mc.items(): fh.write(f"\\newcommand{{\\{kk}}}{{{v}}}\n")
