"""Modelos y espacios finitos de búsqueda de la Práctica 4.

Todos los estimadores reciben el DataFrame original. Las transformaciones se
ajustan al llamar a ``fit`` dentro de cada partición, nunca sobre todo el CSV.
Los espacios usan listas: ParameterSampler muestrea sin reemplazo.
"""

from __future__ import annotations

from math import prod
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder,
)
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from .bases import (
    academic_features, preprocesador_bank, preprocesador_desercion, sumar_uno,
)


SEMILLA = 2026


def _pipeline(problema, clasificador, *, escalar=False, rich=True):
    if problema == "desercion":
        features = (
            FunctionTransformer(academic_features, kw_args={"mode": "rich"})
            if rich else "passthrough"
        )
        pre = preprocesador_desercion(escalar=escalar, disperso=False)
    elif problema == "bank":
        features = "passthrough"
        pre = preprocesador_bank(escalar=escalar, disperso=False)
    else:
        raise ValueError("problema debe ser 'desercion' o 'bank'.")
    return Pipeline([("features", features), ("pre", pre), ("modelo", clasificador)])


def espacios_ensambles(problema: str) -> dict:
    """Cuatro familias; cada espacio contiene mucho más de 50 combinaciones.

    LightGBM y CatBoost se importan aquí para permitir inspeccionar el resto del
    módulo sin instalar esas bibliotecas. No se activa early stopping con el
    pliegue externo de prueba. Todos los ajustes usan un hilo por estimador.
    """
    if problema not in {"desercion", "bank"}:
        raise ValueError("problema debe ser 'desercion' o 'bank'.")
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    bosque = {
        "modelo__n_estimators": [100, 200, 400],
        "modelo__max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
        "modelo__max_depth": [None, 5, 8, 12, 20],
        "modelo__min_samples_leaf": [1, 2, 4, 8, 16],
        "modelo__criterion": ["gini", "entropy", "log_loss"],
        "modelo__class_weight": [None, "balanced", "balanced_subsample"],
    }
    extra = {**bosque, "modelo__bootstrap": [False, True]}
    lightgbm = {
        "modelo__n_estimators": [100, 200, 400],
        "modelo__learning_rate": [0.02, 0.04, 0.07, 0.1],
        "modelo__num_leaves": [7, 15, 31, 63],
        "modelo__max_depth": [-1, 4, 6, 8],
        "modelo__reg_lambda": [0.0, 0.5, 2.0, 5.0, 10.0],
        "modelo__min_child_samples": [10, 20, 40, 80],
        "modelo__subsample": [0.6, 0.8, 1.0],
        "modelo__colsample_bytree": [0.6, 0.8, 1.0],
        "modelo__class_weight": [None, "balanced"],
    }
    catboost = {
        "modelo__iterations": [100, 200, 400],
        "modelo__learning_rate": [0.02, 0.04, 0.07, 0.1],
        "modelo__depth": [3, 4, 5, 6, 7],
        "modelo__l2_leaf_reg": [1.0, 3.0, 5.0, 10.0, 20.0],
        "modelo__subsample": [0.6, 0.8, 1.0],
        "modelo__rsm": [0.6, 0.8, 1.0],
        # 'None' es el valor del enum de CatBoost, no una clase del dataset.
        "modelo__auto_class_weights": ["None", "Balanced", "SqrtBalanced"],
    }
    if problema == "desercion":
        # La ingeniería determinista también se compara dentro de la búsqueda.
        for espacio in (bosque, extra, lightgbm, catboost):
            espacio["features"] = [
                "passthrough",
                FunctionTransformer(academic_features, kw_args={"mode": "rich"}),
            ]
    else:
        for espacio in (bosque, extra, lightgbm, catboost):
            espacio["pre__cat__codificar"] = [
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ]
    modelos = {
        "Random Forest": (
            _pipeline(problema, RandomForestClassifier(random_state=SEMILLA, n_jobs=1)),
            bosque,
        ),
        "Extra-Trees": (
            _pipeline(problema, ExtraTreesClassifier(random_state=SEMILLA, n_jobs=1)),
            extra,
        ),
        "LightGBM": (
            _pipeline(problema, LGBMClassifier(
                random_state=SEMILLA, n_jobs=1, verbosity=-1,
                subsample_freq=1, deterministic=True, force_col_wise=True,
            )),
            lightgbm,
        ),
        "CatBoost": (
            _pipeline(problema, CatBoostClassifier(
                random_seed=SEMILLA, thread_count=1, verbose=False,
                allow_writing_files=False, bootstrap_type="Bernoulli",
            )),
            catboost,
        ),
    }
    assert all(prod(len(v) for v in p.values()) >= 50 for _, p in modelos.values())
    return modelos


def bases_bank() -> dict:
    """Referencias simples, con y sin pesos, para los 41,188 registros."""
    return {
        "Dummy": _pipeline("bank", DummyClassifier(strategy="most_frequent")),
        "Árbol": _pipeline("bank", DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=10, random_state=SEMILLA,
        )),
        "Naive Bayes": _pipeline("bank", GaussianNB(var_smoothing=1e-8)),
        "kNN": _pipeline("bank", KNeighborsClassifier(
            n_neighbors=21, weights="distance", n_jobs=1,
        ), escalar=True),
        "Regresión logística": _pipeline("bank", LogisticRegression(
            max_iter=4000, random_state=SEMILLA,
        ), escalar=True),
        "Regresión logística balanceada": _pipeline("bank", LogisticRegression(
            class_weight="balanced", max_iter=4000, random_state=SEMILLA,
        ), escalar=True),
    }


def _pipeline_categorico_bank(clasificador, *, n_bins=4, strategy="quantile"):
    """Conserva CategoricalNB: todas sus entradas deben ser enteros no negativos.

    El código cero queda reservado para categorías desconocidas. Las magnitudes
    de Bank se discretizan dentro de cada fit; no se pasan números negativos ni
    se inventa un orden continuo entre las categorías.
    """
    pre = Pipeline([
        ("columnas", ColumnTransformer([
            ("cat", Pipeline([
                ("imputar", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("codificar", OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64,
                )),
            ]), make_column_selector(dtype_exclude=np.number)),
            ("num", Pipeline([
                ("imputar", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("discretizar", KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal", strategy=strategy,
                    quantile_method="averaged_inverted_cdf", subsample=None,
                    random_state=SEMILLA,
                )),
            ]), make_column_selector(dtype_include=np.number)),
        ], sparse_threshold=0)),
        ("reservar_desconocido", FunctionTransformer(sumar_uno)),
    ])
    return Pipeline([("features", "passthrough"), ("pre", pre), ("modelo", clasificador)])


def _clasificador_sin_pesos_del_otro_problema(estimador):
    """Clona hiperparámetros y elimina pesos ligados a etiquetas de deserción."""
    nuevo = clone(estimador)
    cambios = {}
    for clave, valor in nuevo.get_params(deep=True).items():
        parametro = clave.rsplit("__", 1)[-1]
        if parametro in {"class_weight", "class_weights"} and isinstance(valor, (dict, list, tuple, np.ndarray)):
            cambios[clave] = None
        elif parametro in {"class_names", "class_prior", "classes_count"} and valor is not None:
            cambios[clave] = None
        elif (parametro in {"loss_function", "objective"} and isinstance(valor, str)
              and valor in {"MultiClass", "multiclass", "multiclassova"}):
            cambios[clave] = "Logloss" if parametro == "loss_function" else "binary"
    if cambios:
        nuevo.set_params(**cambios)
        warnings.warn(
            "Transferencia a Bank: se descartan pesos/nombres de las clases de "
            "deserción y se adapta el objetivo binario: " + ", ".join(cambios),
            UserWarning,
            stacklevel=2,
        )
    return nuevo


def _adaptar_bank(estimador):
    if hasattr(estimador, "best_estimator_"):
        estimador = estimador.best_estimator_
    if isinstance(estimador, Pipeline):
        # Conservar escalado cuando lo requiere el clasificador; nunca conservar
        # columnas académicas ni transformadores ajustados del otro dataset.
        ultimo = estimador.steps[-1][1]
        if isinstance(ultimo, CategoricalNB):
            parametros = estimador.get_params(deep=True)
            prefijo = "pre__columnas__num__discretizar__"
            return _pipeline_categorico_bank(
                _clasificador_sin_pesos_del_otro_problema(ultimo),
                n_bins=parametros.get(prefijo + "n_bins", 4),
                strategy=parametros.get(prefijo + "strategy", "quantile"),
            )
        return _adaptar_bank(estimador.steps[-1][1])
    if isinstance(estimador, (VotingClassifier, StackingClassifier)):
        nuevo = clone(estimador)
        nuevo.set_params(estimators=[
            (nombre, _adaptar_bank(base)) if base != "drop" else (nombre, "drop")
            for nombre, base in estimador.estimators
        ])
        if isinstance(estimador, StackingClassifier) and estimador.final_estimator is not None:
            nuevo.set_params(final_estimator=_clasificador_sin_pesos_del_otro_problema(
                estimador.final_estimator,
            ))
        return nuevo
    if isinstance(estimador, CalibratedClassifierCV):
        nuevo = clone(estimador)
        nuevo.set_params(estimator=_adaptar_bank(estimador.estimator))
        return nuevo
    nuevo = _clasificador_sin_pesos_del_otro_problema(estimador)
    if isinstance(nuevo, CategoricalNB):
        return _pipeline_categorico_bank(nuevo)
    escalar = isinstance(nuevo, (LogisticRegression, SVC, LinearSVC, KNeighborsClassifier))
    return _pipeline("bank", nuevo, escalar=escalar)


def adaptar_ganador_bank(nombre: str, fitted_estimator):
    """Transfiere familia e hiperparámetros, nunca los parámetros ya ajustados.

    Se reconstruye el preprocesamiento con columnas de Bank. Los pesos
    automáticos (``balanced`` y equivalentes) se recalculan al entrenar; los
    diccionarios/listas para las tres clases de deserción se descartan con aviso.
    También admite que el ganador sea votación, stacking o un clasificador base.
    El nombre sirve como dato descriptivo, no para inferir el tipo de modelo.
    """
    if not isinstance(nombre, str) or not nombre:
        raise ValueError("nombre debe identificar al ganador de deserción.")
    return _adaptar_bank(fitted_estimator)
