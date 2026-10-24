"""¿Qué cambia en el sistema mientras el litio se ralentiza? Diagnóstico de la ralentización dentro de cada corrida de 50 ns.

segmentos.py muestra que el primer segmento de 5 ns de cada corrida larga es más rápido que los tardíos (25–50 ns). Aquí se
mide, en los mismos segmentos consecutivos de 5 ns, qué más cambia, para acotar la causa:
  movilidad   end-point de Li, del N del TFSI y de los átomos del polímero guardados en data/iones (posiciones − centro de
              masas). Los átomos del polímero son los de los tipos polymer_solvate_types de HTP-MD, una lista fija que no
              coincide con los átomos que coordinan al Li en cada sistema (a menudo son H o C), pero sí son átomos del
              polímero: sirven para medir la movilidad de la matriz, no su coordinación con el Li.
  coordinación  número medio de N de TFSI en torno a cada Li (corte en el primer mínimo de g(r) Li–N de cada sistema).
  energía     energía potencial y presión medias del segmento (nVT_averages.txt, medias de 100 ps), por átomo; la producción
              es NVT, así que el volumen no cambia.
  agregados   aggregates.txt (clústeres Li–O(TFSI) calculados por LAMMPS en cada frame): fracción de Li sin ningún O de TFSI
              en su clúster y tamaño medio del clúster de un Li.
Lo local sale de data/iones/<id>.npz; energía y agregados se leen en remoto del zip de HTP-MD (unos MB por trayectoria).
Incremental: un JSON por trayectoria en resultados/ralentizacion/; sin argumentos agrega lo que haya.
Memoria: una trayectoria cada vez y solo los frames necesarios del polímero (máximo ~1,1 GB en float32).
Uso: nice -n 19 python ralentizacion.py --calcular   (reanudable)   ·   python ralentizacion.py   → resultados/ralentizacion.json
"""
import glob, io, json, os, sys, time, zipfile
import numpy as np
import pandas as pd
from scipy import stats

DEST = os.environ.get("HTPMD_DEST", "data/iones")
CACHE = "resultados/ralentizacion"
SEG = 2500                     # frames por segmento de 5 ns (un frame cada 2 ps)
N_SEG, TARDIOS, N_BOOT, SEMILLA = 10, 5, 2000, 20261021


def rdf_minimo(a, b, L, rmax=8.0, nb=80):
    """Primer mínimo de g(r) tras el primer pico (por encima de 1,5 Å); a, b: (frames, n, 3)."""
    h = np.zeros(nb)
    for fa, fb in zip(a, b):
        d = fa[:, None, :] - fb[None, :, :]; d -= L * np.round(d / L)
        h += np.histogram(np.linalg.norm(d, axis=-1).ravel(), bins=nb, range=(0, rmax))[0]
    c = (np.arange(nb) + 0.5) * rmax / nb; g = h / (c ** 2)
    k = 15 + np.argmax(g[15:]); return float(c[k + np.argmin(g[k:k + 25])])


def locales(i):
    z = np.load(f"{DEST}/{i}.npz")
    L = np.float32(z["caja"][0]); com = z["com"].astype(np.float32)[:, None, :]
    li = z["li"] - com; an = z["tfsi"] - com
    bordes = np.arange(0, N_SEG * SEG + 1, SEG)
    pol = z["polimero"]; pb = pol[bordes] - com[bordes] if pol.shape[1] else None; del pol
    dep = lambda r, s: float(np.mean(np.sum((r[(s + 1) * SEG] - r[s * SEG]) ** 2, -1)) / 30.0)   # Å²/ns, ventana de 5 ns
    rc = rdf_minimo(li[::500], an[::500], L)
    out = {"n_li": int(li.shape[1]), "n_pol_guardados": int(pb.shape[1]) if pb is not None else 0, "corte_li_n": rc, "segmentos": []}
    for s in range(N_SEG):
        cn = []
        for f in range(s * SEG, (s + 1) * SEG, 25):
            d = li[f][:, None, :] - an[f][None, :, :]; d -= L * np.round(d / L)
            cn.append((np.linalg.norm(d, axis=-1) < rc).sum(1).mean())
        out["segmentos"].append({"D_li": dep(li, s), "D_tfsi": dep(an, s),
                                 "D_matriz": float(np.mean(np.sum((pb[s + 1] - pb[s]) ** 2, -1)) / 30.0) if pb is not None else None,
                                 "cn_li_n": float(np.mean(cn))})
    return out


def remotos(i):
    from htpmd_remoto import Remoto
    z = zipfile.ZipFile(io.BufferedReader(Remoto(i), 16 << 20))
    n_at = int(next(l for l in z.read("relaxed.lmp").decode().splitlines() if l.strip().endswith("atoms")).split()[0])
    t = np.loadtxt(io.BytesIO(z.read("nVT_averages.txt")))          # paso, tiempo (fs), T, P, PE, KE, Etot
    pasos, libres, tam = [], [], []
    with z.open("aggregates.txt") as fh:
        f = io.TextIOWrapper(fh, encoding="ascii")
        while True:
            l = f.readline()
            if not l: break
            if not l.startswith("ITEM: TIMESTEP"): continue
            pasos.append(int(f.readline())); f.readline(); n = int(f.readline())
            for _ in range(5): f.readline()
            a = np.loadtxt(io.StringIO("".join(f.readline() for _ in range(n))), dtype=int, usecols=(1, 2), ndmin=2)
            es_li = a[:, 0] == 90; cl_li = a[es_li, 1]; cl_o = a[~es_li, 1]
            libres.append(float(np.mean(~np.isin(cl_li, cl_o))))
            u, cnt = np.unique(a[:, 1], return_counts=True); tam.append(float(cnt[np.isin(u, cl_li)].mean()))
    pasos, libres, tam = np.array(pasos), np.array(libres), np.array(tam)
    seg_t = (t[:, 1] - 1e-3) // 5e6; seg_a = (pasos * 2e-6) // 5                       # fs → segmento de 5 ns; paso de 2 fs
    out = {"n_atomos": n_at, "segmentos": []}
    for s in range(N_SEG):
        k, j = seg_t == s, seg_a == s
        out["segmentos"].append({"pe_por_atomo": float(t[k, 4].mean() / n_at), "presion": float(t[k, 3].mean()),
                                 "pe_sd_100ps_por_atomo": float(t[k, 4].std(ddof=1) / n_at),
                                 "li_libres": float(libres[j].mean()), "tam_cluster_li": float(tam[j].mean())})
    return out


def calcular():
    os.makedirs(CACHE, exist_ok=True)
    T = pd.read_csv("resultados/tabla_por_trayectoria.csv")
    ids = T[T.ns == 50].id.astype(int).tolist()
    for n, i in enumerate(ids, 1):
        f = f"{CACHE}/{i}.json"
        if os.path.exists(f): continue
        t0 = time.time()
        try:
            r = {"id": i, "local": locales(i), "remoto": remotos(i)}
            json.dump(r, open(f + ".tmp", "w"), indent=1); os.replace(f + ".tmp", f)
            print(f"[{n}/{len(ids)}] {i}: {time.time() - t0:.0f} s", flush=True)
        except Exception as e:
            print(f"[{n}/{len(ids)}] {i}: ERROR {e!r}", flush=True)


def agregar():
    T = pd.read_csv("resultados/tabla_por_trayectoria.csv"); lit = T.set_index("id").literatura
    filas = []
    for f in sorted(glob.glob(f"{CACHE}/*.json")):
        r = json.load(open(f)); L, Rm = r["local"]["segmentos"], r["remoto"]["segmentos"]
        g = lambda v: np.exp(np.mean(np.log(v)))                                      # media geométrica
        fila = {"id": r["id"], "literatura": bool(lit[r["id"]])}
        for k in ("D_li", "D_tfsi", "D_matriz"):
            v = [x[k] for x in L]
            fila[f"{k}_primero_tardios"] = v[0] / g(v[TARDIOS:]) if None not in v and min(v) > 0 else np.nan
        for k in ("cn_li_n",):
            v = [x[k] for x in L]; fila[f"{k}_cambio"] = v[0] - np.mean(v[TARDIOS:])
        for k in ("pe_por_atomo", "presion", "li_libres", "tam_cluster_li"):
            v = [x[k] for x in Rm]; fila[f"{k}_cambio"] = v[0] - np.mean(v[TARDIOS:])    # primero − tardíos
        fila["pe_ruido_por_atomo"] = float(np.mean([x["pe_sd_100ps_por_atomo"] for x in Rm[TARDIOS:]]))
        filas.append(fila)
    D = pd.DataFrame(filas); rng = np.random.default_rng(SEMILLA)
    ic = lambda b: [float(np.nanquantile(b, .025)), float(np.nanquantile(b, .975))]
    R = {"n": len(D), "por_variable": {}, "correlaciones_con_D_li": {}}
    for c in [c for c in D.columns if c.endswith(("_primero_tardios", "_cambio"))]:
        x = D[c].dropna().to_numpy(); razon = c.endswith("_primero_tardios")
        est = (lambda v: float(np.exp(np.median(np.log(v))))) if razon else (lambda v: float(np.median(v)))
        R["por_variable"][c] = {"n": len(x), "mediana": est(x), "ic95": ic([est(rng.choice(x, len(x))) for _ in range(N_BOOT)]),
                                "frac_mayor": float(np.mean(x > (1 if razon else 0))),
                                "wilcoxon_p": float(stats.wilcoxon(np.log(x) if razon else x).pvalue),
                                "cribado": est(D.loc[~D.literatura, c].dropna().to_numpy()),
                                "literatura": est(D.loc[D.literatura, c].dropna().to_numpy())}
    y = np.log(D["D_li_primero_tardios"])
    for c in [c for c in D.columns if (c.endswith(("_primero_tardios", "_cambio"))) and c != "D_li_primero_tardios"]:
        ok = D[c].notna() & y.notna(); a, b = y[ok].to_numpy(), D.loc[ok, c].to_numpy()
        bs = [stats.spearmanr(a[j], b[j])[0] for j in (rng.integers(0, len(a), len(a)) for _ in range(N_BOOT))]
        R["correlaciones_con_D_li"][c] = {"n": int(ok.sum()), "spearman": float(stats.spearmanr(a, b)[0]), "ic95": ic(bs),
                                          "p": float(stats.spearmanr(a, b)[1])}
    R["pe_ruido_por_atomo_mediana"] = float(D.pe_ruido_por_atomo.median())
    R["pe_cambio_sobre_ruido_mediana"] = float(np.median(D.pe_por_atomo_cambio / D.pe_ruido_por_atomo))
    json.dump(R, open("resultados/ralentizacion.json", "w"), indent=1)
    D.to_csv("resultados/ralentizacion_por_sistema.csv", index=False)
    escribir_tex(R)
    print(f"{len(D)} trayectorias")
    for c, v in R["por_variable"].items():
        print(f"  {c:28s} mediana {v['mediana']:+.4g} {[round(u, 4) for u in v['ic95']]}  >ref {v['frac_mayor']:.0%}  p {v['wilcoxon_p']:.1e}  "
              f"cribado {v['cribado']:+.4g} literatura {v['literatura']:+.4g}")
    print("  correlación (Spearman) con la ralentización del Li:")
    for c, v in R["correlaciones_con_D_li"].items():
        print(f"    {c:28s} {v['spearman']:+.2f} {[round(u, 2) for u in v['ic95']]}  (n = {v['n']})")


def escribir_tex(R):
    """Macros (paper/numeros_ralentizacion.tex) y tabla de la información suplementaria (paper/tabla_ralentizacion.tex)."""
    V, C = R["por_variable"], R["correlaciones_con_D_li"]
    menos = lambda t: t.replace("-", "$-$")                               # signo menos tipográfico
    f2 = lambda x: menos(f"{x:.2f}"); pc = lambda x: f"{100 * x:.0f}"
    nom = {"D_li_primero_tardios": "Li", "D_tfsi_primero_tardios": "Tfsi", "D_matriz_primero_tardios": "Mat",
           "pe_por_atomo_cambio": "Pe", "presion_cambio": "Press", "tam_cluster_li_cambio": "Cluster",
           "li_libres_cambio": "Free", "cn_li_n_cambio": "Cn"}
    fmt = {"Li": f2, "Tfsi": f2, "Mat": f2, "Pe": lambda x: menos(f"{x:.3f}"), "Press": lambda x: menos(f"{x:.0f}"),
           "Cluster": f2, "Free": f2, "Cn": f2}
    M = {"slowN": str(R["n"]), "slowPeOverNoise": f"{R['pe_cambio_sobre_ruido_mediana']:.1f}"}
    for c, n in nom.items():
        v = V[c]; g = fmt[n]
        M[f"slow{n}"] = g(v["mediana"]); M[f"slow{n}Lo"], M[f"slow{n}Hi"] = (g(u) for u in v["ic95"])
        M[f"slow{n}FracPct"] = pc(v["frac_mayor"]); M[f"slow{n}FracBelowPct"] = pc(1 - v["frac_mayor"]); M[f"slow{n}N"] = str(v["n"])
        M[f"slow{n}Screen"] = g(v["cribado"]); M[f"slow{n}Lit"] = g(v["literatura"])
        if c in C:
            M[f"slowCorr{n}"] = f2(C[c]["spearman"]); M[f"slowCorr{n}Lo"], M[f"slowCorr{n}Hi"] = (f2(u) for u in C[c]["ic95"])
    S = json.load(open("resultados/segmentos.json"))["extremos"]          # el Li, con el mismo intervalo que el texto principal
    M["slowLiLo"], M["slowLiHi"] = (f2(u) for u in S["primero_vs_tardios_ic95"])
    with open("paper/numeros_ralentizacion.tex", "w") as fh:
        fh.write("% Generado por ralentizacion.py a partir de resultados/ralentizacion.json. No editar a mano.\n")
        for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    filas = [("Li", "Lithium diffusivity (ratio)", "slower"), ("Tfsi", "Anion diffusivity (ratio)", "slower"),
             ("Mat", "Polymer-atom mobility (ratio)", "slower"),
             ("Pe", "Potential energy (kcal\\,mol$^{-1}$ per atom)", "lower"), ("Press", "Pressure (atm)", "lower"),
             ("Cluster", "Size of the cluster of a Li (ions)", "smaller"), ("Free", "Fraction of Li without anion", "larger"),
             ("Cn", "Anion N around each Li", "fewer")]
    with open("paper/tabla_ralentizacion.tex", "w") as fh:
        fh.write("% Generado por ralentizacion.py. No editar a mano.\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
                 "Quantity & First vs late (95\\% CI) & Late segments & Systems & Screening & Literature & "
                 "$\\rho$ with Li slowdown (95\\% CI) \\\\\n\\midrule\n")
        for n, etiqueta, sentido in filas:
            pct = M[f"slow{n}FracBelowPct"] if n == "Free" else M[f"slow{n}FracPct"]
            rho = "n/a" if n == "Li" else f"{M[f'slowCorr{n}']} ({M[f'slowCorr{n}Lo']} to {M[f'slowCorr{n}Hi']})"
            fh.write(f"{etiqueta} & {M[f'slow{n}']} ({M[f'slow{n}Lo']} to {M[f'slow{n}Hi']}) & {sentido} & {pct}\\% & "
                     f"{M[f'slow{n}Screen']} & {M[f'slow{n}Lit']} & {rho} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    if "--calcular" in sys.argv: calcular()
    agregar()
