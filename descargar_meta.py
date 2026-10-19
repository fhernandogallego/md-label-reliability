"""Lee el meta.json (SMILES del monómero y del polímero, sal, temperatura...) de cada trayectoria de HTP-MD SIN descargar el zip:
solo el índice del zip y el fichero pequeño, con peticiones Range (~200 KB por muestra en vez de 0,8–3 GB).
Secuencial (el servicio de URLs firmadas se atasca con peticiones simultáneas) y reanudable: un JSON por muestra en data/meta/.
Uso: python descargar_meta.py            (todas las de data/molality-vs-conductivity.csv)
     python descargar_meta.py 0 3            (trozo 0 de 3; lanzar 0, 1 y 2 a la vez para ir 3 veces más rápido)
     nohup python descargar_meta.py > data/meta.log 2>&1 &    (en segundo plano; tarda unas horas)
"""
import csv, io, json, os, sys, time, zipfile
from htpmd_remoto import Remoto

D = "data/meta"; os.makedirs(D, exist_ok=True)
ids = [r["Sample ID"] for r in csv.DictReader(open("data/molality-vs-conductivity.csv"))]
if len(sys.argv) == 3:                                  # reparto en trozos: python descargar_meta.py <k> <n> procesa los ids con índice % n == k
    k, nt = int(sys.argv[1]), int(sys.argv[2]); ids = [x for j, x in enumerate(ids) if j % nt == k]
hechas = errores = 0; t0 = time.time()
for n, i in enumerate(ids, 1):
    f = f"{D}/{i}.json"
    if os.path.exists(f): hechas += 1; continue
    for intento in range(3):
        try:
            z = zipfile.ZipFile(io.BufferedReader(Remoto(i), 1 << 16))     # búfer pequeño: solo se leen el índice y meta.json
            meta = json.loads(z.read("meta.json")); meta["_ficheros"] = z.namelist()
            json.dump(meta, open(f + ".tmp", "w")); os.replace(f + ".tmp", f); hechas += 1; break
        except Exception as e:
            err = repr(e); time.sleep(10 * (intento + 1))
    else:
        errores += 1; print(f"{i}: ERROR {err}", flush=True)
    if n % 200 == 0:
        print(f"{n}/{len(ids)} hechas={hechas} errores={errores} ({(time.time() - t0) / 60:.0f} min)", flush=True)
print(f"fin: {hechas} de {len(ids)}, {errores} errores", flush=True)
