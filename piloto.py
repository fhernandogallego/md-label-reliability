"""Recalcula el piloto SCAYLE y sus intervalos desde las etiquetas por corrida."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comun import N_BOOT, SEMILLA_BOOT, ic

MODOS = ('htpmd_original', 'htpmd_replica', 'reproducir', 'vel1', 'vel2', 'rec1', 'rec2')


def validar_etiquetas(a, tabla):
    """No admitir corridas incompletas, controles ajenos o modos de otras campañas."""
    required = {'sistema', 'modo', 'extremos', 'medio', 'T_ns'}
    if not required <= set(a) or a.empty or a[list(required)].isna().any().any():
        raise ValueError('Faltan columnas o valores en las etiquetas del piloto')
    if a.duplicated(['sistema', 'modo']).any():
        raise ValueError('Etiquetas duplicadas de sistema/modo')
    if any(set(g.modo) != set(MODOS) for _, g in a.groupby('sistema')):
        raise ValueError('Cada sistema requiere cinco corridas y dos controles HTP-MD')
    if not np.isfinite(a[['extremos', 'medio', 'T_ns']].to_numpy(float)).all():
        raise ValueError('Etiquetas no finitas')
    if not np.allclose(a.T_ns, 5.0, rtol=0, atol=1e-10):
        raise ValueError('Todas las producciones del piloto deben durar 5 ns')
    if a.sistema.nunique() < 2:
        raise ValueError('Se requieren al menos dos sistemas para el bootstrap')
    cortas = tabla[tabla.ns.eq(5)]
    for modo, replica in [('htpmd_original', False), ('htpmd_replica', True)]:
        controles = a[a.modo.eq(modo)].set_index('sistema')
        fuente = cortas[cortas.carpeta.str.contains('new_config', na=False).eq(replica)]
        fuente = fuente[fuente.sistema.isin(controles.index)]
        if fuente.sistema.duplicated().any() or set(fuente.sistema) != set(controles.index):
            raise ValueError('Controles HTP-MD ausentes o ambiguos')
        fuente = fuente.set_index('sistema').loc[controles.index]
        for col, original in [('extremos', 'D_extremos'), ('medio', 'D_medio')]:
            if not np.allclose(controles[col], np.log(fuente[original]), rtol=0, atol=1e-12):
                raise ValueError(f'Control {modo}/{col} distinto de la tabla HTP-MD')


def calcular(a, n_boot=N_BOOT, semilla=SEMILLA_BOOT):
    """Bootstrap pareado por sistemas; mismo orden y semilla que comparar.py."""
    r = {'n_sistemas': int(a.sistema.nunique()),
         'corridas': int((~a.modo.str.startswith('htpmd_')).sum())}
    for est in ['extremos', 'medio']:
        p = a.pivot(index='sistema', columns='modo', values=est).sort_index()
        d = p.reproducir - p.htpmd_original
        r[est] = {
            'reproducir_vs_original': {'n': len(p), 'razon_mediana': float(np.exp(d.median())),
                                      'sd_dif_sobre_raiz2': float(d.std(ddof=1) / np.sqrt(2))},
            'sd_dinamica': float(np.sqrt(p[['htpmd_original', 'reproducir', 'vel1', 'vel2']].var(axis=1, ddof=1).mean())),
            'n_dinamica': len(p),
            'sd_total_recocido': float(np.sqrt(p[['htpmd_original', 'rec1', 'rec2']].var(axis=1, ddof=1).mean())),
            'n_total': len(p),
            'sd_total_htpmd_mismos_sistemas': float((p.htpmd_original - p.htpmd_replica).std(ddof=1) / np.sqrt(2)),
        }
    p = a.pivot(index='sistema', columns='modo', values='extremos').sort_index()
    # Resampling system contributions keeps all modes of each system together.
    contrib = {
        'dif_reproducir': (p.reproducir - p.htpmd_original).to_numpy(),
        'dif_recocido': (p[['rec1', 'rec2']].mean(axis=1) - p.htpmd_original).to_numpy(),
        'sd_dinamica': p[['htpmd_original', 'reproducir', 'vel1', 'vel2']].var(axis=1, ddof=1).to_numpy(),
        'sd_entre_recocidas': p[['rec1', 'rec2']].var(axis=1, ddof=1).to_numpy(),
        'sd_replicas_htpmd': ((p.htpmd_original - p.htpmd_replica).to_numpy() ** 2 / 2),
    }
    rng = np.random.default_rng(semilla)
    intervals = {}
    for key, values in contrib.items():
        indices = rng.integers(0, len(p), size=(n_boot, len(p)))
        transform = np.sqrt if key.startswith('sd_') else lambda x: x
        boot = transform(values[indices].mean(axis=1))
        intervals[key] = {'valor': float(transform(values.mean())), 'ic95': ic(boot)}
    r['intervalos_extremos'] = intervals
    return r


def desglose(a, n_boot=N_BOOT, semilla=SEMILLA_BOOT):
    """Dispersión de cada tipo de corrida por separado (etiqueta end-point) y parte estructural en estos sistemas.
    No forma parte del resumen archivado: se guarda aparte para no alterar su esquema verificado."""
    p = a.pivot(index='sistema', columns='modo', values='extremos').sort_index()
    v = lambda cols: p[cols].var(axis=1, ddof=1).to_numpy()            # varianza dentro de cada sistema
    contrib = {
        'sd_repeticion_exacta': v(['htpmd_original', 'reproducir']),   # misma estructura y mismas velocidades
        'sd_velocidades_nuevas': v(['vel1', 'vel2']),                  # misma estructura, velocidades nuevas
        'sd_misma_estructura': v(['htpmd_original', 'reproducir', 'vel1', 'vel2']),
        'sd_estructuras_distintas': v(['htpmd_original', 'htpmd_replica', 'rec1', 'rec2']),
    }
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(p), size=(n_boot, len(p)))               # mismos remuestreos para todas las medidas
    r = {k: {'valor': float(np.sqrt(x.mean())), 'ic95': ic(np.sqrt(x[idx].mean(axis=1)))} for k, x in contrib.items()}
    misma, distintas = contrib['sd_misma_estructura'], contrib['sd_estructuras_distintas']
    parte = lambda i: 1 - misma[i].mean(axis=-1) / distintas[i].mean(axis=-1)
    r['parte_estructural'] = {'valor': float(parte(np.arange(len(p)))), 'ic95': ic(parte(idx))}
    return r


def macros_desglose(d):
    m = {'pilSdExact': d['sd_repeticion_exacta'], 'pilSdNewVel': d['sd_velocidades_nuevas'],
         'pilSdDistinct': d['sd_estructuras_distintas']}
    out = {}
    for name, row in m.items():
        for suffix, value in [('', row['valor']), ('Lo', row['ic95'][0]), ('Hi', row['ic95'][1])]:
            out[f'{name}{suffix}'] = f'{value:.2f}'
    s = d['parte_estructural']
    for suffix, value in [('', s['valor']), ('Lo', s['ic95'][0]), ('Hi', s['ic95'][1])]:
        out[f'pilStrPct{suffix}'] = f'{100 * value:.0f}'
    return '% Generated by piloto.py from per-run labels: dispersion by run type.\n' + ''.join(
        f'\\newcommand{{\\{key}}}{{{value}}}\n' for key, value in out.items())


def diferencias(actual, reference, path=''):
    """Diferencias numéricas de todos los campos; rechaza cambios de esquema."""
    if isinstance(reference, dict):
        if set(actual) != set(reference):
            raise ValueError(f'Esquema distinto en {path}')
        return [d for key in reference for d in diferencias(actual[key], reference[key], f'{path}/{key}')]
    if isinstance(reference, list):
        if len(actual) != len(reference):
            raise ValueError(f'Longitud distinta en {path}')
        return [d for x, y in zip(actual, reference) for d in diferencias(x, y, path)]
    return [abs(float(actual) - float(reference))]


def macros(r):
    m = {'pilN': str(r['n_sistemas']), 'pilRuns': str(r['corridas']),
         'pilRepRatio': f"{r['extremos']['reproducir_vs_original']['razon_mediana']:.2f}"}
    for key, name, transform in [('dif_reproducir', 'RepShift', np.exp), ('dif_recocido', 'RecShift', np.exp),
                                 ('sd_dinamica', 'SdDyn', float), ('sd_entre_recocidas', 'SdRec', float),
                                 ('sd_replicas_htpmd', 'SdHtp', float)]:
        row = r['intervalos_extremos'][key]
        for suffix, value in [('', row['valor']), ('Lo', row['ic95'][0]), ('Hi', row['ic95'][1])]:
            m[f'pil{name}{suffix}'] = f'{transform(value):.2f}'
    return '% Generated by piloto.py from per-run labels; system bootstrap recomputed.\n' + ''.join(
        f'\\newcommand{{\\{key}}}{{{value}}}\n' for key, value in m.items())


def main():
    root = Path('resultados')
    a = pd.read_csv(root / 'etiquetas_piloto.csv')
    tabla = pd.read_csv(root / 'tabla_por_trayectoria.csv')
    validar_etiquetas(a, tabla)
    provenance = json.loads((root / 'piloto_scayle_procedencia.json').read_text())
    if (provenance['bootstrap_resamples'], provenance['bootstrap_seed']) != (N_BOOT, SEMILLA_BOOT):
        raise ValueError('Configuración bootstrap distinta de la procedencia publicada')
    for filename, field in [('etiquetas_piloto.csv', 'source_labels_sha256'),
                            ('piloto_scayle_referencia.json', 'source_summary_sha256')]:
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != provenance[field]:
            raise ValueError(f'Hash de procedencia incorrecto: {filename}')
    r = calcular(a)
    if (r['corridas'], r['n_sistemas']) != (provenance['completed_runs'], provenance['systems']):
        raise ValueError('Recuento del piloto distinto de la procedencia publicada')
    reference = json.loads((root / 'piloto_scayle_referencia.json').read_text())
    delta = max(diferencias(r, reference))
    if delta > 1e-12:
        raise ValueError(f'El recálculo difiere del resumen de los autores: {delta}')
    (root / 'piloto_scayle.json').write_text(json.dumps(r, indent=2) + '\n')
    Path('paper/numeros_piloto.tex').write_text(macros(r))
    d = desglose(a)
    (root / 'piloto_desglose.json').write_text(json.dumps(d, indent=2) + '\n')
    Path('paper/numeros_piloto_desglose.tex').write_text(macros_desglose(d))
    print(f"Piloto recalculado: {r['corridas']} corridas / {r['n_sistemas']} sistemas, "
          f'{N_BOOT} bootstrap por sistemas; diferencia máxima con referencia {delta:.3g}.')


if __name__ == '__main__':
    main()
