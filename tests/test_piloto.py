"""Audita el bootstrap contra el análisis original y rechaza entradas inválidas."""
import json
from pathlib import Path
import unittest

import pandas as pd

from piloto import calcular, diferencias, validar_etiquetas

ROOT = Path(__file__).resolve().parents[1]


class PilotoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = pd.read_csv(ROOT / 'resultados/etiquetas_piloto.csv')
        cls.table = pd.read_csv(ROOT / 'resultados/tabla_por_trayectoria.csv')

    def test_original_analysis_and_row_order(self):
        validar_etiquetas(self.labels, self.table)
        reference = json.loads((ROOT / 'resultados/piloto_scayle_referencia.json').read_text())
        self.assertLess(max(diferencias(calcular(self.labels), reference)), 1e-12)
        shuffled = self.labels.sample(frac=1, random_state=4)
        self.assertEqual(calcular(self.labels), calcular(shuffled))

    def test_missing_or_duplicate_run_rejected(self):
        for labels in [self.labels.iloc[1:], pd.concat([self.labels, self.labels.iloc[:1]])]:
            with self.subTest(rows=len(labels)), self.assertRaises(ValueError):
                validar_etiquetas(labels, self.table)

    def test_wrong_control_rejected(self):
        labels = self.labels.copy()
        labels.loc[labels.modo.eq('htpmd_original'), 'extremos'] += .01
        with self.assertRaisesRegex(ValueError, 'Control'):
            validar_etiquetas(labels, self.table)

    def test_wrong_duration_rejected(self):
        labels = self.labels.copy()
        labels.loc[0, 'T_ns'] = 4.9
        with self.assertRaisesRegex(ValueError, '5 ns'):
            validar_etiquetas(labels, self.table)


if __name__ == '__main__':
    unittest.main()
