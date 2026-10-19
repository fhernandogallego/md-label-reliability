"""Tabla compacta de estructuras para el aprendizaje automático (data/ no se sube al repositorio).
Une data/meta/<id>.json (descargar_meta.py) con el identificador de sistema de resultados/tabla_por_trayectoria.csv.
Salida: resultados/estructuras.csv con id, sistema, mol_smiles (monómero) y poly_smiles (polímero).
Uso: python estructuras.py
"""
import glob, json, os
import pandas as pd

T = pd.read_csv("resultados/tabla_por_trayectoria.csv", usecols=["id", "sistema"])
filas = []
for f in glob.glob("data/meta/*.json"):
    m = json.load(open(f))
    filas.append({"id": int(os.path.basename(f)[:-5]), "mol_smiles": m.get("mol_smiles"), "poly_smiles": m.get("poly_smiles")})
E = T.merge(pd.DataFrame(filas), on="id", how="left").sort_values("id")
E.to_csv("resultados/estructuras.csv", index=False)
falta = E.mol_smiles.isna().sum()
print(f"{len(E)} trayectorias; con SMILES {len(E) - falta}; sin SMILES {falta}; sistemas distintos {E.sistema.nunique()}; "
      f"monómeros distintos {E.mol_smiles.nunique()}")
