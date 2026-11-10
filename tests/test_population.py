"""Casos límite de la población que define el denominador de fiabilidad."""
import unittest
import pandas as pd
from comun import originales_cribado

class PopulationTests(unittest.TestCase):
    def test_excludes_literature_long_runs_and_replica_only_systems(self):
        T=pd.DataFrame([
            (2,5,'a','original',False),
            (3,5,'a','new_config_rand_100',False),
            (4,5,'b','new_config_rand_100',False),
            (5,50,'c','original',False),
            (6,5,'d','literature',True),
            (1,5,'a','original',False),
        ],columns=['id','ns','sistema','carpeta','literatura'])
        result=originales_cribado(T)
        self.assertEqual(result.id.tolist(),[1])
        self.assertEqual(result.sistema.tolist(),['a'])
        pd.testing.assert_frame_equal(result,originales_cribado(T.iloc[::-1]))

if __name__=='__main__':unittest.main()
