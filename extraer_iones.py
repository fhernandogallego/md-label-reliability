"""Extrae en streaming (sin guardar el zip de 3 GB) lo que el paper necesita de cada trayectoria cruda de HTP-MD.

Deja data/iones/<id>.npz con posiciones desenvueltas (Å, float32) de:
  li        (frames, nLi, 3)      catión (tipo LAMMPS = cation_raw_type)
  tfsi      (frames, nTFSI, 3)    nitrógeno del anión (anion_raw_type)
  polimero  (frames, nPol, 3)     átomos del polímero que solvatan al Li (polymer_solvate_types)
  com       (frames, 3)           centro de masas de todo el sistema: HTP-MD resta esta deriva antes de calcular el MSD
y además: caja (Å), dt_ps, pasos, temperatura, poblacion (matriz de clústeres iónicos) y meta (JSON).
Las posiciones se guardan SIN corregir; para analizar, usar posición − com.

Uso:  python extraer_iones.py <id> [<id> ...]
      python extraer_iones.py --largas [--trabajos 3]      las 133 de 50 ns; reanudable (salta las ya hechas)
Registro en <destino>/extraccion.log. Variables: HTPMD_DEST (carpeta de salida), HTPMD_MIN_LIBRE_GB (freno de disco, 40 por defecto).
"""
import csv, io, json, os, shutil, sys, time, zipfile
import numpy as np
from itertools import islice
from multiprocessing import Pool
from htpmd_remoto import Remoto, url

DEST = os.environ.get("HTPMD_DEST", "data/iones")          # p. ej. HTPMD_DEST=/media/usuario/DISCO/iones para escribir en un disco externo
MIN_LIBRE_GB = float(os.environ.get("HTPMD_MIN_LIBRE_GB", "40"))   # freno: no empieza otra trayectoria si queda menos espacio libre

def _extraer(i):
    salida = f"{DEST}/{i}.npz"
    z = zipfile.ZipFile(io.BufferedReader(Remoto(i), 32 << 20))
    meta = json.loads(z.read("meta.json"))
    tc, ta = str(meta["cation_raw_type"]), str(meta["anion_raw_type"]); tp = {str(t) for t in meta.get("polymer_solvate_types", [])}
    quiero = {tc, ta} | tp
    li, an, pol, com, pasos, caja = [], [], [], [], [], None
    tc_i, ta_i, tp_i = int(tc), int(ta), np.array(sorted(int(t) for t in tp))
    with z.open("traj.lammpstrj") as fh:
        f = io.TextIOWrapper(fh, encoding="ascii", errors="replace")
        while True:
            l = f.readline()
            if not l: break
            if not l.startswith("ITEM: TIMESTEP"): continue
            pasos.append(int(f.readline())); f.readline(); n = int(f.readline()); f.readline()
            L = np.array([float(f.readline().split()[1]) for _ in range(3)]); caja = L if caja is None else caja
            cols = f.readline().split()[2:]; k = {c: cols.index(c) for c in ("id", "type", "mass", "x", "y", "z", "ix", "iy", "iz")}
            a = np.fromstring("".join(islice(f, n)), sep=" ").reshape(n, len(cols))          # todo el frame de una vez (rápido)
            tipo = a[:, k["type"]].astype(int); masa = a[:, k["mass"]]
            r = a[:, [k["x"], k["y"], k["z"]]] + a[:, [k["ix"], k["iy"], k["iz"]]] * L       # coordenadas desenvueltas
            com.append((masa[:, None] * r).sum(0) / masa.sum())                             # centro de masas de TODO el sistema (deriva)
            orden = np.argsort(a[:, k["id"]], kind="stable"); tipo, r = tipo[orden], r[orden]
            li.append(r[tipo == tc_i].astype(np.float32)); an.append(r[tipo == ta_i].astype(np.float32))
            pol.append(r[np.isin(tipo, tp_i)].astype(np.float32))
    nombres = set(z.namelist())
    poblacion = np.loadtxt(io.BytesIO(z.read("population.txt"))) if "population.txt" in nombres else np.zeros((0, 0))
    tmp = salida + ".parcial.npz"
    np.savez_compressed(tmp, li=np.stack(li), tfsi=np.stack(an), polimero=np.stack(pol), com=np.array(com), caja=np.array(caja), dt_ps=float(meta["time_step"]),
                        pasos=np.array(pasos), temperatura=float(meta["temperature"]), poblacion=poblacion, meta=json.dumps(meta))
    os.replace(tmp, salida)                                                # el .npz solo aparece cuando está completo
    return f"{len(pasos)} frames, {li[0].shape[0]} Li, {an[0].shape[0]} TFSI, {pol[0].shape[0]} at. polímero, {os.path.getsize(salida) / 1e6:.0f} MB"

def extraer(i):
    os.makedirs(DEST, exist_ok=True)
    if os.path.exists(f"{DEST}/{i}.npz"): return i, "ya estaba"
    libre = shutil.disk_usage(DEST).free / 1e9
    if libre < MIN_LIBRE_GB: return i, f"OMITIDA: quedan {libre:.0f} GB libres (< {MIN_LIBRE_GB:.0f} GB)"
    t0 = time.time()
    for intento in range(3):
        try:
            r = _extraer(i); msg = f"{r}, {time.time() - t0:.0f} s"; break
        except Exception as e:
            msg = f"ERROR {e!r}"; time.sleep(30)
    with open(f"{DEST}/extraccion.log", "a") as fh: fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {i}: {msg}\n")
    return i, msg

if __name__ == "__main__":
    args = sys.argv[1:]; trabajos = 1
    if "--trabajos" in args:
        j = args.index("--trabajos"); trabajos = int(args[j + 1]); args = args[:j] + args[j + 2:]
    if args and args[0] == "--largas":
        ids = [r["id"] for r in csv.DictReader(open("resultados/tabla_por_trayectoria.csv")) if float(r["ns"]) == 50.0]
    else:
        ids = args
    with Pool(trabajos) as pool:
        for n, (i, msg) in enumerate(pool.imap_unordered(extraer, ids), 1):
            print(f"[{n}/{len(ids)}] {i}: {msg}", flush=True)
