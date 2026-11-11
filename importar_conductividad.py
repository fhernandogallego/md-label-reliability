"""Importa el CSV exportado por HTP-MD a una entrada compacta redistribuible.
Uso: python importar_conductividad.py /ruta/molality-vs-conductivity.csv
No convierte unidades ni reconstruye valores a partir de correlaciones.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('csv',type=Path)
args=parser.parse_args()
raw=args.csv.read_bytes()
D=pd.read_csv(args.csv).rename(columns={'Sample ID':'id','Li-ion Conductivity (S/cm)':'cond'})
D=D[['id','cond']]
T=pd.read_csv('resultados/tabla_por_trayectoria.csv',usecols=['id'])
D=T.merge(D,on='id',how='left',validate='one_to_one')
if not np.isfinite(D.cond).all() or (D.cond<=0).any():
    raise ValueError('Se requieren conductividades positivas y finitas para todos los IDs del artículo')
D.to_csv('resultados/conductividad.csv',index=False)
json.dump({'source':'HTP-MD user-interface CSV export', 'source_filename':args.csv.name,
           'source_sha256':hashlib.sha256(raw).hexdigest(),'rows':len(D),
           'source_url':'https://www.htpmd.matr.io/',
           'license':'CC BY-NC 4.0',
           'license_evidence':'https://data.matr.io/htp/',
           'license_url':'https://creativecommons.org/licenses/by-nc/4.0/',
           'units':'S/cm, as reported by the source; no conversion applied'},
          open('resultados/conductividad_procedencia.json','w'),indent=2)
print(f'Importadas {len(D)} conductividades a resultados/conductividad.csv')
