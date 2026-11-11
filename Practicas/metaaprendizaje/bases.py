"""Pipelines fijas de P3 y combinaciones probabilísticas sin fugas de ajuste."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .datos import _directorio_practicas


SEED_P3 = 42
SEED = 2026
CLASES_DESERCION = ["dropout", "enrolled", "graduate"]
NOMINALES_DESERCION = {
    "Marital status", "Application mode", "Course", "Daytime/evening attendance",
    "Previous qualification", "Nacionality", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date",
    "Gender", "Scholarship holder", "International",
}


def nominales_desercion(df):
    """Selecciona códigos nominales conforme al esquema semántico de P3."""
    return [c for c in df if c in NOMINALES_DESERCION]


def numericas_desercion(df):
    """Selecciona magnitudes y atributos derivados en el orden de entrada."""
    return [c for c in df if c not in NOMINALES_DESERCION]


def academic_features(df, mode="rich"):
    """Reproduce las tasas académicas de P3 sin consultar datos de otras filas."""
    result = df.copy()
    if mode == "none":
        return result
    for sem in ["1st", "2nd"]:
        prefix = f"Curricular units {sem} sem"
        if f"{prefix} (enrolled)" not in result:
            continue
        enrolled = result[f"{prefix} (enrolled)"]
        denominator = enrolled.replace(0, np.nan)
        approved = result[f"{prefix} (approved)"]
        evaluations = result[f"{prefix} (evaluations)"]
        without = result[f"{prefix} (without evaluations)"]
        result[f"{sem}_approved_ratio"] = approved / denominator
        result[f"{sem}_evaluations_ratio"] = evaluations / denominator
        result[f"{sem}_without_evaluations_ratio"] = without / denominator
        result[f"{sem}_no_enrollment"] = enrolled.eq(0).astype(int)
        if mode == "rich":
            result[f"{sem}_credited_ratio"] = result[f"{prefix} (credited)"] / denominator
            result[f"{sem}_not_approved"] = enrolled - approved
            result[f"{sem}_grade_times_progress"] = result[f"{prefix} (grade)"] * approved / denominator
    return result


def sumar_uno(a):
    """Reserva el cero para nominales no vistos por CategoricalNB."""
    return np.asarray(a, dtype=np.int64) + 1


def preprocesador_desercion(escalar=False, disperso=False):
    """Reproduce imputación, one-hot y escalado del preprocesador final de P3."""
    categorico = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=disperso)),
    ])
    numerico = [("imputar", SimpleImputer(strategy="median", keep_empty_features=True))]
    if escalar:
        numerico.append(("escalar", StandardScaler()))
    return ColumnTransformer([
        ("cat", categorico, nominales_desercion),
        ("num", Pipeline(numerico), numericas_desercion),
    ], remainder="drop", sparse_threshold=1.0 if disperso else 0)


def _preprocesador_categorico():
    """Ordinales y discretización de la variante CategoricalNB ganadora de P3."""
    return Pipeline([
        ("columnas", ColumnTransformer([
            ("cat", Pipeline([
                ("imputar", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("codificar", OrdinalEncoder(handle_unknown="use_encoded_value",
                                             unknown_value=-1, dtype=np.int64)),
            ]), nominales_desercion),
            ("num", Pipeline([
                ("imputar", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("discretizar", KBinsDiscretizer(
                    n_bins=5, encode="ordinal", strategy="quantile",
                    quantile_method="averaged_inverted_cdf", subsample=None,
                    random_state=SEED_P3)),
            ]), numericas_desercion),
        ], sparse_threshold=0)),
        ("reservar_desconocido", FunctionTransformer(sumar_uno)),
    ])


def bases_desercion(base: Path):
    """Reconstruye los cinco finales del JSON de P3, sin eval ni optimización.

    Solo se leen parámetros escalares o diccionarios; las representaciones
    textuales de estimadores y funciones nunca se ejecutan.
    """
    ruta = _directorio_practicas(base) / "resultados_practica_03" / "mejores_parametros.json"
    configuraciones = json.loads(ruta.read_text(encoding="utf-8"))
    nombres = ["Árbol", "Naive Bayes", "kNN", "SVM", "Regresión logística"]
    if set(configuraciones) != set(nombres):
        raise ValueError("El JSON de P3 no contiene exactamente los cinco modelos finales.")
    componentes = {
        "Árbol": (preprocesador_desercion(),
                  DecisionTreeClassifier(random_state=SEED_P3)),
        "Naive Bayes": (_preprocesador_categorico(), CategoricalNB()),
        "kNN": (preprocesador_desercion(escalar=True),
                KNeighborsClassifier(algorithm="brute", n_jobs=1)),
        "SVM": (preprocesador_desercion(escalar=True),
                SVC(kernel="rbf", cache_size=256, random_state=SEED_P3)),
        "Regresión logística": (preprocesador_desercion(escalar=True, disperso=True),
            LogisticRegression(solver="lbfgs", l1_ratio=0.0, max_iter=10000,
                               tol=1e-4, random_state=SEED_P3)),
    }
    resultado = {}
    for nombre in nombres:
        config = configuraciones[nombre]
        pre, modelo = componentes[nombre]
        features = (FunctionTransformer(academic_features)
                    if nombre in {"SVM", "Regresión logística"} else "passthrough")
        estimador = Pipeline([("features", features), ("pre", pre), ("modelo", modelo)])
        parametros = {k: v for k, v in config.items() if "__" in k}
        # La única representación textual que requiere reconstrucción es el escalador.
        if "pre__num__escalar" in parametros:
            if parametros["pre__num__escalar"] != "StandardScaler()":
                raise ValueError(f"El escalador de {nombre} difiere del final auditado de P3.")
            parametros["pre__num__escalar"] = StandardScaler()
        estimador.set_params(**parametros)
        resultado[nombre] = estimador
    return resultado


def _bases_con_etiquetas_codificadas(base: Path):
    """Adapta pesos porque Voting y Stacking entrenan con y codificada por sklearn."""
    resultado = bases_desercion(base)
    for estimador in resultado.values():
        modelo = estimador.named_steps["modelo"]
        pesos = modelo.get_params().get("class_weight")
        if isinstance(pesos, dict):
            modelo.set_params(class_weight={i: pesos[clase]
                                           for i, clase in enumerate(CLASES_DESERCION)})
    return resultado


def ensambles_desercion(base: Path):
    """Crea votaciones y stacking con calibración interna de la Pipeline SVM."""
    bases = _bases_con_etiquetas_codificadas(base)
    claves = {"Árbol": "arbol", "Naive Bayes": "nb", "kNN": "knn",
              "SVM": "svm", "Regresión logística": "logistica"}
    duros = [(claves[n], clone(est)) for n, est in bases.items()]
    probabilisticos = [(claves[n],
        CalibratedClassifierCV(
            estimator=clone(est), method="sigmoid",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED), n_jobs=1)
        if n == "SVM" else clone(est)) for n, est in bases.items()]
    return {
        "Votación dura": VotingClassifier(estimators=duros, voting="hard", n_jobs=1),
        "Votación suave": VotingClassifier(estimators=clone(probabilisticos, safe=False),
                                           voting="soft", n_jobs=1),
        "Stacking logístico": StackingClassifier(
            estimators=clone(probabilisticos, safe=False),
            final_estimator=LogisticRegression(max_iter=10000, random_state=SEED),
            stack_method="predict_proba",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
            n_jobs=1),
    }


def preprocesador_bank(escalar=False, disperso=False, ordinal=False):
    """Imputa/codifica por dtype dentro del modelo; escala numéricas si se pide.

    Ordinal permite árboles con menos columnas; el desconocido se codifica -1.
    El modo one-hot genera una matriz densa salvo disperso=True.
    """
    codificar = (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                 if ordinal else OneHotEncoder(handle_unknown="ignore", sparse_output=disperso))
    categorico = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("codificar", codificar),
    ])
    numerico = [("imputar", SimpleImputer(strategy="median", keep_empty_features=True))]
    if escalar:
        numerico.append(("escalar", StandardScaler()))
    return ColumnTransformer([
        ("cat", categorico, make_column_selector(dtype_exclude=np.number)),
        ("num", Pipeline(numerico), make_column_selector(dtype_include=np.number)),
    ], remainder="drop", sparse_threshold=1.0 if disperso and not ordinal else 0)
