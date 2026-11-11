"""Comprobaciones breves de datos, aislamiento CV, búsquedas y serialización.

Ejecutar desde la raíz: .venv-practicas/Scripts/python Practicas/tests_practica_04.py
No ejecuta las búsquedas de 50 configuraciones ni el notebook completo.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from metaaprendizaje.bases import bases_desercion, ensambles_desercion
from metaaprendizaje.datos import cargar_bank, cargar_desercion
from metaaprendizaje.evaluacion import evaluar, jsonable
from metaaprendizaje.modelos import adaptar_ganador_bank, bases_bank, espacios_ensambles


BASE = Path(__file__).resolve().parent


def muestra(X, y, cantidad=240):
    """Muestra pequeña estratificada reproducible, conservando todas las clases."""
    indices, _ = train_test_split(np.arange(len(y)), train_size=cantidad,
                                 stratify=y, random_state=2026)
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)


class Practica04Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Xd, cls.yd, cls.cvd, cls.ids = cargar_desercion(BASE)
        cls.Xb, cls.yb = cargar_bank(BASE)

    def test_pliegues_originales_cubren_exactamente_una_vez(self):
        original = pd.read_csv(BASE / "resultados_practica_03" / "pliegues.csv",
                               dtype={"id": "string"}).set_index("id")
        cobertura = np.zeros(len(self.yd), dtype=int)
        generados = list(StratifiedKFold(5, shuffle=True, random_state=42).split(self.Xd, self.yd))
        for i, ((tr, va), (tr_p3, va_p3)) in enumerate(zip(self.cvd, generados), 1):
            self.assertEqual(len(np.intersect1d(tr, va)), 0)
            self.assertEqual(len(tr) + len(va), len(self.yd))
            np.testing.assert_array_equal(tr, tr_p3)
            np.testing.assert_array_equal(va, va_p3)
            self.assertTrue(original.loc[self.ids.iloc[va], "pliegue_externo"].eq(i).all())
            cobertura[va] += 1
        np.testing.assert_array_equal(cobertura, np.ones(len(self.yd), dtype=int))

    def test_bank_X_es_original_sin_duration_ni_objetivo(self):
        original = pd.read_csv(BASE / "datos" / "bank-additional-full.csv", sep=";")
        self.assertEqual(original.shape, (41188, 21))
        assert_frame_equal(self.Xb, original.drop(columns=["duration", "y"]))
        assert_series_equal(self.yb, original.y.map({"yes": 1, "no": 0}).astype("int64"))
        self.assertEqual(self.Xb.shape, (41188, 19))

    def test_ensambles_calibran_pipeline_y_remapean_pesos(self):
        ensambles = ensambles_desercion(BASE)
        for nombre, ensamble in ensambles.items():
            clone(ensamble)
            for clave, estimador in ensamble.estimators:
                if clave == "svm" and nombre != "Votación dura":
                    self.assertIsInstance(estimador, CalibratedClassifierCV)
                    self.assertIsInstance(estimador.estimator, Pipeline)
                    self.assertEqual(estimador.cv.n_splits, 3)
                    estimador = estimador.estimator
                pesos = estimador.named_steps["modelo"].get_params().get("class_weight")
                if isinstance(pesos, dict):
                    self.assertEqual(set(pesos), {0, 1, 2})
                pre = estimador.named_steps["pre"]
                if isinstance(pre, Pipeline):
                    pre = pre.named_steps["columnas"]
                with self.assertRaises(NotFittedError):
                    check_is_fitted(pre)

    def test_cada_preprocesador_aprende_solo_de_su_entrenamiento(self):
        X = pd.DataFrame({"age": [float(i) for i in range(30)],
                          "job": ["unica" if i == 0 else "comun" for i in range(30)]})
        X.loc[0, "age"] = 1000000.0
        y = pd.Series([0, 1] * 15)
        cv = list(StratifiedKFold(3, shuffle=True, random_state=2026).split(X, y))
        modelo = bases_bank()["Regresión logística"]
        self.assertFalse(hasattr(modelo.named_steps["pre"], "transformers_"))
        with threadpool_limits(limits=1):
            resultados = cross_validate(modelo, X, y, cv=cv, scoring="f1_macro",
                                        return_estimator=True, error_score="raise", n_jobs=1)
        for estimador, (tr, va) in zip(resultados["estimator"], cv):
            pre = estimador.named_steps["pre"]
            media = pre.named_transformers_["num"].named_steps["escalar"].mean_[0]
            self.assertAlmostEqual(media, X.iloc[tr].age.mean())
            categorias = pre.named_transformers_["cat"].named_steps["codificar"].categories_[0]
            if 0 in va:
                self.assertNotIn("unica", categorias)
        self.assertFalse(hasattr(modelo.named_steps["pre"], "transformers_"))

    def test_espacios_tienen_50_configuraciones_distintas_y_validas(self):
        for problema in ("desercion", "bank"):
            for nombre, (estimador, espacio) in espacios_ensambles(problema).items():
                with self.subTest(problema=problema, modelo=nombre):
                    parametros = list(ParameterSampler(espacio, n_iter=50, random_state=2026))
                    firmas = {json.dumps(jsonable(p), sort_keys=True) for p in parametros}
                    self.assertEqual(len(firmas), 50)
                    for p in parametros:
                        clone(estimador).set_params(**p)
                    self.assertIn("modelo__max_features" if nombre in {"Random Forest", "Extra-Trees"}
                                  else "modelo__learning_rate", espacio)

    def test_ambas_bibliotecas_boosting_ajustan_en_ambos_problemas(self):
        with threadpool_limits(limits=1):
            for problema, X, y in (("desercion", self.Xd, self.yd), ("bank", self.Xb, self.yb)):
                Xs, ys = muestra(X, y)
                tr, va = train_test_split(np.arange(len(ys)), test_size=.25,
                                         stratify=ys, random_state=2026)
                for nombre, (estimador, espacio) in espacios_ensambles(problema).items():
                    if nombre not in {"LightGBM", "CatBoost"}:
                        continue
                    with self.subTest(problema=problema, modelo=nombre):
                        parametros = next(iter(ParameterSampler(espacio, n_iter=1, random_state=2026)))
                        modelo = clone(estimador).set_params(**parametros)
                        modelo.fit(Xs.iloc[tr], ys.iloc[tr])
                        pred = np.asarray(modelo.predict(Xs.iloc[va])).ravel()
                        self.assertEqual(len(pred), len(va))
                        self.assertTrue(set(pred) <= set(ys))

    def test_transferir_ensamble_a_bank_admite_numericas_negativas(self):
        X, y = muestra(self.Xb, self.yb, cantidad=180)
        self.assertLess(X["cons.conf.idx"].min(), 0)
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Transferencia a Bank.*")
            warnings.filterwarnings("ignore", message="Bins whose width.*")
            modelo = adaptar_ganador_bank("Votación dura", ensambles_desercion(BASE)["Votación dura"])
            modelo.fit(X, y)
            self.assertEqual(len(modelo.predict(X.iloc[:10])), 10)

    def test_cache_conserva_scores_y_predicciones_oof(self):
        X = pd.DataFrame({"age": np.arange(40, dtype=float), "job": ["a", "b"] * 20})
        y = pd.Series([0, 1] * 20)
        cv = StratifiedKFold(5, shuffle=True, random_state=2026)
        with tempfile.TemporaryDirectory(prefix="practica04_cache_") as temporal:
            salida = Path(temporal) / "resultados"
            cache = Path(temporal) / "cache"
            primero = evaluar("prueba", bases_bank()["Regresión logística"], X, y, cv,
                               1, salida, cache, n_jobs=1)
            for i, (_, va) in enumerate(cv.split(X, y)):
                esperado = f1_score(y.iloc[va], primero["oof"][va], average="macro")
                self.assertAlmostEqual(primero["pliegues"].iloc[i].f1_macro, esperado)
            with patch("metaaprendizaje.evaluacion.cross_validate",
                       side_effect=AssertionError("La caché válida no debe recalcular")):
                segundo = evaluar("prueba", bases_bank()["Regresión logística"], X, y, cv,
                                   1, salida, cache, n_jobs=1)
            assert_frame_equal(primero["pliegues"], segundo["pliegues"])
            np.testing.assert_array_equal(primero["oof"], segundo["oof"])

    def test_joblib_se_carga_en_proceso_nuevo_con_funciones_importables(self):
        X, y = muestra(self.Xd, self.yd, cantidad=180)
        modelo = bases_desercion(BASE)["Regresión logística"]
        with threadpool_limits(limits=1):
            modelo.fit(X.iloc[:150], y.iloc[:150])
            esperado = modelo.predict(X.iloc[150:]).tolist()
        with tempfile.TemporaryDirectory(prefix="practica04_joblib_") as temporal:
            carpeta = Path(temporal)
            joblib.dump(modelo, carpeta / "modelo.joblib")
            X.iloc[150:].to_csv(carpeta / "X.csv", index=False)
            codigo = (
                "import json,sys,joblib,pandas as pd; from pathlib import Path; "
                "p=Path(sys.argv[1]); m=joblib.load(p/'modelo.joblib'); "
                "print(json.dumps(m.predict(pd.read_csv(p/'X.csv')).tolist()))"
            )
            entorno = os.environ.copy()
            entorno["PYTHONPATH"] = str(BASE) + os.pathsep + entorno.get("PYTHONPATH", "")
            proceso = subprocess.run([sys.executable, "-c", codigo, str(carpeta)],
                                     cwd=BASE.parent, env=entorno, capture_output=True,
                                     text=True, check=True, timeout=60)
            self.assertEqual(json.loads(proceso.stdout.strip()), esperado)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Bins whose width.*")
    warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
    unittest.main(verbosity=2)
