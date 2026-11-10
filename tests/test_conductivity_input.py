"""La entrada debe conservar IDs, unidades y cobertura; nunca rellenar datos ausentes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pandas as pd

IMPORTER=Path(__file__).resolve().parents[1]/'importar_conductividad.py'

class ConductivityInputTests(unittest.TestCase):
    def run_import(self,folder,ids,values):
        root=Path(folder)
        (root/'resultados').mkdir()
        pd.DataFrame({'id':[8,3]}).to_csv(root/'resultados/tabla_por_trayectoria.csv',index=False)
        pd.DataFrame({'Sample ID':ids,'Li-ion Conductivity (S/cm)':values}).to_csv(root/'source.csv',index=False)
        return subprocess.run([sys.executable,str(IMPORTER),str(root/'source.csv')],cwd=root,capture_output=True,text=True)

    def test_aligns_ids_and_keeps_reported_values(self):
        with tempfile.TemporaryDirectory() as folder:
            result=self.run_import(folder,[3,8,99],[.004,.002,.1])
            self.assertEqual(result.returncode,0,result.stderr)
            root=Path(folder)/'resultados'
            output=pd.read_csv(root/'conductividad.csv')
            self.assertEqual(output.id.tolist(),[8,3])
            self.assertEqual(output.cond.tolist(),[.002,.004])
            provenance=json.loads((root/'conductividad_procedencia.json').read_text())
            self.assertEqual(provenance['rows'],2)
            self.assertEqual(len(provenance['source_sha256']),64)

    def test_rejects_missing_duplicate_or_nonpositive_labels(self):
        for ids,values in [([3],[.004]),([3,8,8],[.004,.002,.003]),([3,8],[.004,0])]:
            with self.subTest(ids=ids,values=values),tempfile.TemporaryDirectory() as folder:
                result=self.run_import(folder,ids,values)
                self.assertNotEqual(result.returncode,0)
                self.assertFalse((Path(folder)/'resultados/conductividad.csv').exists())

if __name__=='__main__':unittest.main()
