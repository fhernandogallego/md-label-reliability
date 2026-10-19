"""Guarda la ruta S3 de cada trayectoria (sin la firma temporal): identifica la formulación y la tanda, y permite emparejar 5 ns con 50 ns.
Secuencial y reanudable (el servicio se atasca con peticiones simultáneas). Uso: python descargar_rutas.py  → data/rutas.csv"""
import csv, json, os, time, urllib.request
ids = [r["Sample ID"] for r in csv.DictReader(open("data/molality-vs-conductivity.csv"))]
hechas = set()
if os.path.exists("data/rutas.csv") and os.path.getsize("data/rutas.csv") > 0:
    hechas = {r["id"] for r in csv.DictReader(open("data/rutas.csv")) if r["ruta"]}
nuevo = not hechas
with open("data/rutas.csv", "w" if nuevo else "a", newline="") as fh:
    w = csv.writer(fh)
    if nuevo: w.writerow(["id", "ruta"])
    for n, i in enumerate(ids, 1):
        if i in hechas: continue
        ruta = ""
        for intento in range(3):
            try:
                r = urllib.request.Request(f"https://www.htpmd.matr.io/api/trajectories/{i}/raw", headers={"User-Agent": "md-label-reliability/0.1 (uso academico)"})
                ruta = json.load(urllib.request.urlopen(r, timeout=15))["s3_presigned_url"].split("?")[0].split(".amazonaws.com/")[1]; break
            except Exception:
                time.sleep(3 + 5 * intento)
        w.writerow([i, ruta]); fh.flush(); time.sleep(0.1)
        if n % 500 == 0: print(n, flush=True)
print("fin")
