"""Descarga de HTP-MD (TRI/MIT) lo ligero de cada trayectoria: condiciones (duración, temperatura) y curvas MSD de Li y TFSI.
API pública de la web https://www.htpmd.matr.io (sin sesión). Reanudable: guarda un JSON por trayectoria en data/curvas/.
Uso: python descargar_curvas.py [hilos=3]"""
import csv, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
API = "https://www.htpmd.matr.io/api"; D = "data/curvas"; os.makedirs(D, exist_ok=True)
ids = [r["Sample ID"] for r in csv.DictReader(open("data/molality-vs-conductivity.csv"))]

def get(p):
    for intento in range(4):
        try:
            r = urllib.request.Request(API + p, headers={"User-Agent": "md-label-reliability/0.1 (uso academico)"})
            return json.load(urllib.request.urlopen(r, timeout=40))
        except Exception as e:
            if getattr(e, "code", None) in (404, 500): return None
            time.sleep(2 + 3 * intento)
    return None

def uno(i):
    f = f"{D}/{i}.json"
    if os.path.exists(f): return 0
    out = {"id": i, "condiciones": (get(f"/trajectories/{i}/conditions") or {}).get("conditions"),
           "li": (get(f"/trajectories/{i}/li-msd-curve") or {}).get("li_msd_curve"),
           "tfsi": (get(f"/trajectories/{i}/tfsi-msd-curve") or {}).get("tfsi_msd_curve")}
    json.dump(out, open(f, "w")); time.sleep(0.15); return 1

with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as ex:
    for n, _ in enumerate(ex.map(uno, ids), 1):
        if n % 500 == 0: print(n, "de", len(ids), flush=True)
print("fin", len(os.listdir(D)))
