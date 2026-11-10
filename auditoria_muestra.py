"""Auditoría descriptiva de disponibilidad y composición de la muestra replicada.
No infiere equivalencia a partir de tests no significativos. Solo necesita tablas publicadas.
"""
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from comun import originales_cribado

T = pd.read_csv('resultados/tabla_por_trayectoria.csv')
E = pd.read_csv('resultados/estructuras.csv')[['id', 'mol_smiles']]
A = originales_cribado(T).merge(E, on='id', validate='one_to_one')
B = T[(T.ns == 5) & T.carpeta.str.contains('new_config', na=False)]
rep = set(B.sistema) & set(A.sistema)
L = set(T[(T.ns == 50) & ~T.literatura].sistema)
mols = [Chem.MolFromSmiles(s) for s in A.mol_smiles]
# Au/Cu son marcadores de conexión de HTP-MD, no átomos del monómero físico.
A['heavy_atoms'] = [sum(a.GetAtomicNum() > 1 and a.GetAtomicNum() not in (29, 79) for a in m.GetAtoms()) for m in mols]
for name, z in [('oxygen',8), ('nitrogen',7), ('fluorine',9)]:
    A[name] = [sum(a.GetAtomicNum() == z for a in m.GetAtoms()) for m in mols]
A['rings'] = [m.GetRingInfo().NumRings() for m in mols]
A['ln_D5'] = np.log(A.D_extremos)
rows = []
for col in ['heavy_atoms','oxygen','nitrogen','fluorine','rings','ln_D5']:
    selected = A.sistema.isin(rep)
    scale = A.loc[~selected,col].std(ddof=1)
    rows.append({'descriptor': col, 'replicated_median': float(A.loc[selected,col].median()),
                 'rest_median': float(A.loc[~selected,col].median()),
                 'standardized_mean_difference': float((A.loc[selected,col].mean()-A.loc[~selected,col].mean())/scale) if scale > 0 else None})
R = {'population': 'one original label per screening system', 'n_population': len(A),
     'nominal_folder_size': 100, 'available_new_configurations': len(B), 'paired_systems': len(rep),
     'new_configurations_without_original': sorted(set(B.sistema)-set(A.sistema)),
     'screening_long_systems': len(L), 'replicated_with_long': len(rep & L), 'descriptors': rows,
     'availability_note': 'The released index contains 95 new-configuration trajectories in the rand_100 folder. One lacks an original run; the index does not explain the other five nominal entries.'}
json.dump(R,open('resultados/auditoria_muestra.json','w'),indent=2,allow_nan=False)
names = {'heavy_atoms':'Heavy atoms (excluding connectors)', 'oxygen':'O atoms', 'nitrogen':'N atoms', 'fluorine':'F atoms', 'rings':'Rings', 'ln_D5':r'$\ln D_{\mathrm{end}}$, 5 ns'}
with open('paper/tabla_muestra.tex','w') as f:
    f.write('\\begin{tabular}{lrrr}\n\\toprule\nDescriptor & Replicated & Rest & Mean difference / SD \\\\\n\\midrule\n')
    for r in rows:
        difference = f"{r['standardized_mean_difference']:+.2f}" if r['standardized_mean_difference'] is not None else "--"
        f.write(f"{names[r['descriptor']]} & {r['replicated_median']:.2f} & {r['rest_median']:.2f} & {difference} \\\\\n")
    f.write('\\bottomrule\n\\end{tabular}\n')
print(json.dumps({k:v for k,v in R.items() if k != 'descriptors'},indent=2))

with open('paper/numeros_muestra.tex','w') as f:
    for key,value in {
        'sampleNominal': R['nominal_folder_size'],
        'sampleAvailable': R['available_new_configurations'],
        'sampleMissing': R['nominal_folder_size'] - R['available_new_configurations'],
        'samplePaired': R['paired_systems'],
    }.items():
        f.write(f"\\newcommand{{\\{key}}}{{{value}}}\n")
