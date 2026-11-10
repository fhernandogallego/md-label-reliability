"""Comprueba consistencia entre análisis independientes y entradas publicadas."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from comun import originales_cribado

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sin-conductividad',action='store_true',help='Auditar solo los análisis que no requieren la entrada de conductividad')
args=parser.parse_args()
def read(name):return json.loads(Path(f'resultados/{name}.json').read_text())
T=pd.read_csv('resultados/tabla_por_trayectoria.csv')
A=originales_cribado(T)
U,H,M,C,V=(read(n) for n in ['incertidumbre','techo','ml_demo','asignacion','validacion_asignacion'])
assert len(A)==U['n_poblacion']==H['n_poblacion']==M['n_sistemas']==C['N']
rep=T[T.ns.eq(5)].groupby('sistema').filter(lambda g:len(g)==2)
assert rep.sistema.nunique()==U['replicas']['n_sistemas']==V['n_replicados']
for col in ['D_extremos','D_medio']:
    g=rep.groupby('sistema')[col]
    noise=np.log(g.first()/g.last()).std(ddof=1)/np.sqrt(2)
    expected=1-noise**2/np.log(A[col]).var(ddof=1)
    assert np.isclose(U['replicas'][col]['fiabilidad'],expected)
    assert np.isclose(H[col]['fiabilidad'],expected)
    assert np.isclose(C[f'modelo_{col}']['fiabilidad'],expected)
assert V['campana']['segundas_corridas']==V['campana']['n']//2
assert 'rerun policy' in V['campana']['bootstrap']
G=read('generalidad')
assert G['n_poblacion']==len(A)
assert np.isclose(G['Li']['fiabilidad'],U['replicas']['D_medio']['fiabilidad'])
for ion in ['Li','TFSI']:
    assert 0<G[ion]['fiabilidad']<1
if not args.sin_conductividad:
    assert G['conductividad_incluida']
    D=pd.read_csv('resultados/conductividad.csv')
    assert set(D.id)==set(T.id) and D.id.is_unique and (D.cond>0).all()
    assert G['n_poblacion']==len(A)
    assert np.isclose(G['Li']['fiabilidad'],U['replicas']['D_medio']['fiabilidad'])
    for ion in ['Li','TFSI','Cond']:
        assert 0<G[ion]['fiabilidad']<1
print(f'Consistencia verificada: {len(A)} sistemas originales de cribado, {rep.sistema.nunique()} pares replicados.')
if args.sin_conductividad: print('Conductividad omitida explícitamente en esta comprobación.')
