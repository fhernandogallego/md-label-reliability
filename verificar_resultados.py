"""Comprueba consistencia entre análisis independientes y entradas publicadas."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from comun import originales_cribado
from piloto import calcular, diferencias, macros, validar_etiquetas

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

P=read('piloto_scayle')
provenance=read('piloto_scayle_procedencia')
for filename, field in [('etiquetas_piloto.csv', 'source_labels_sha256'),
                        ('piloto_scayle_referencia.json', 'source_summary_sha256')]:
    assert hashlib.sha256(Path('resultados', filename).read_bytes()).hexdigest()==provenance[field]
assert P['corridas']==provenance['completed_runs'] and P['n_sistemas']==provenance['systems']
assert provenance['per_run_inputs_included']
labels=pd.read_csv('resultados/etiquetas_piloto.csv')
validar_etiquetas(labels,T)
replicas=set(T[T.ns.eq(5) & T.carpeta.str.contains('new_config',na=False)].sistema)
largas=set(T[T.ns.eq(50) & ~T.literatura].sistema)
assert len(replicas & largas)==provenance['eligible_systems']
selected=np.random.default_rng(provenance['system_selection_seed']).choice(
    sorted(replicas & largas),provenance['systems'],replace=False)
assert set(labels.sistema)==set(selected)
recalculated=calcular(labels)
assert max(diferencias(recalculated,P))<1e-12
assert max(diferencias(P,read('piloto_scayle_referencia')))<1e-12
assert Path('paper/numeros_piloto.tex').read_text()==macros(P)
from piloto import desglose, macros_desglose
D=desglose(labels)
assert max(diferencias(D,read('piloto_desglose')))<1e-12
assert Path('paper/numeros_piloto_desglose.tex').read_text()==macros_desglose(D)
print('Piloto: 50 corridas, 20 controles, selección aleatoria, bootstrap y macros verificados desde las etiquetas.')

# Ninguna macro sin valor ("n/a") puede aparecer en la prosa del artículo o del suplementario
import re
sin_valor={m for f in Path('paper').glob('numeros*.tex')
           for m in re.findall(r'\\newcommand\{\\(\w+)\}\{n/a\}',f.read_text())}
for doc in ['paper/jcs.tex','paper/si.tex']:
    usadas=sin_valor & set(re.findall(r'\\(\w+)',Path(doc).read_text()))
    assert not usadas,f'{doc} usa macros sin valor: {sorted(usadas)}'
print('Texto: ninguna macro sin valor (n/a) en jcs.tex ni si.tex.')

# R1/R2: con ρ = 1 el objetivo largo reproduce la simulación principal (dentro del error Monte Carlo)
OL=read('objetivo_largo'); C0=read('asignacion')
for top in ['0.1','0.01']:
    e=OL['campanas'][f'uno|{top}']
    for b in ['1.0','4.0']:
        ref=C0['resultados'][f'D_extremos|normal|{top}|uniforme|{b}']['recall']
        assert abs(e['uniforme'][b]-ref)<0.03,(top,b,e['uniforme'][b],ref)
assert 0<OL['rho']['valor']<=1.05 and OL['potencia']['0.1']['potencia_5pct']<0.5
print(f"Objetivo largo: ρ = {OL['rho']['valor']:.2f}; con ρ = 1 reproduce la simulación principal; potencia de la campaña real {OL['potencia']['0.1']['potencia_5pct']:.2f}.")

# Ampliación: mismos sistemas que el piloto, controles iguales a la tabla de HTP-MD y macros coherentes con el JSON
EA=pd.read_csv('resultados/etiquetas_ampliacion.csv')
assert set(EA.sistema)<=set(labels.sistema)
cortas5=T[T.ns.eq(5)]
for modo,replica in [('htpmd_original',False),('htpmd_replica',True)]:
    ctl=EA[EA.modo.eq(modo)].set_index('sistema')
    src=cortas5[cortas5.carpeta.str.contains('new_config',na=False).eq(replica)].set_index('sistema').loc[ctl.index]
    assert np.allclose(ctl.extremos,np.log(src.D_extremos),atol=1e-12) and np.allclose(ctl.medio,np.log(src.D_medio),atol=1e-12)
AM=read('ampliacion')
macros_amp=Path('paper/numeros_ampliacion.tex').read_text()
assert f"\\newcommand{{\\ampTN}}{{{AM['extremos']['T393']['razon_D_393_353']['n_sistemas']}}}" in macros_amp
assert f"\\newcommand{{\\ampEndTRatio}}{{{AM['extremos']['T393']['razon_D_393_353']['valor']:.1f}}}" in macros_amp
print(f"Ampliación: {EA.sistema.nunique()} sistemas del piloto, controles HTP-MD verificados, macros coherentes.")
