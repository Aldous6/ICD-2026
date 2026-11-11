"""Curvas por pliegue con warm_start y staged_predict, sin compartir ajustes."""

from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .bases import preprocesador_desercion
from .evaluacion import firma


CANTIDADES = [1, 5, 10, 20, 50, 100, 200, 300, 400]


def referencias():
    pre = preprocesador_desercion()
    arbol = DecisionTreeClassifier(max_depth=None, random_state=2026)
    tocon = DecisionTreeClassifier(max_depth=1, random_state=2026)
    return {
        'Árbol sin límite': Pipeline([('pre', clone(pre)), ('modelo', clone(arbol))]),
        'Bagging 400': Pipeline([('pre', clone(pre)), ('modelo', BaggingClassifier(
            estimator=clone(arbol), n_estimators=400, bootstrap=True,
            warm_start=True, random_state=2026, n_jobs=1))]),
        'Tocón': Pipeline([('pre', clone(pre)), ('modelo', clone(tocon))]),
        'AdaBoost 400': Pipeline([('pre', clone(pre)), ('modelo', AdaBoostClassifier(
            estimator=clone(tocon), n_estimators=400, learning_rate=1., random_state=2026))]),
    }


def curvas_sesgo_varianza(X, y, cv, carpeta, cache, reutilizar=True):
    carpeta, cache = Path(carpeta), Path(cache)
    carpeta.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    clave = firma(X, y, cv, CANTIDADES, referencias())
    archivo = cache / 'curvas.joblib'
    if reutilizar and archivo.exists() and all((carpeta / nombre).exists() for nombre in
            ['curvas_pliegues.csv', 'curvas_resumen.csv', 'curvas_sesgo_varianza.png']):
        anterior = joblib.load(archivo)
        if anterior['firma'] == clave:
            print('Curvas: resultados comprobados de caché', flush=True)
            return anterior['tabla']
    filas = []
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(cv, 1):
            print(f'Curvas: pliegue {fold}/5', flush=True)
            inicio_pre = perf_counter()
            pre = preprocesador_desercion()
            Xt = pre.fit_transform(X.iloc[tr], y.iloc[tr])
            Xv = pre.transform(X.iloc[va])
            tiempo_pre = perf_counter() - inicio_pre
            bag = referencias()['Bagging 400'].named_steps['modelo']
            tiempo = tiempo_pre
            for n in CANTIDADES:
                bag.set_params(n_estimators=n)
                inicio = perf_counter()
                bag.fit(Xt, y.iloc[tr])
                tiempo += perf_counter() - inicio
                pv, pt = bag.predict(Xv), bag.predict(Xt)
                filas.append(_fila('Bagging', fold, n, n, y.iloc[va], pv,
                                   y.iloc[tr], pt, tiempo))
            ada = referencias()['AdaBoost 400'].named_steps['modelo']
            inicio = perf_counter()
            ada.fit(Xt, y.iloc[tr])
            tiempo = tiempo_pre + perf_counter() - inicio
            efectivos = len(ada.estimators_)
            for ronda, (pv, pt) in enumerate(zip(ada.staged_predict(Xv),
                                                 ada.staged_predict(Xt)), 1):
                if ronda in CANTIDADES or ronda == efectivos:
                    filas.append(_fila('AdaBoost', fold, ronda, efectivos,
                                       y.iloc[va], pv, y.iloc[tr], pt, tiempo))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(carpeta / 'curvas_pliegues.csv', index=False)
    resumen = tabla.groupby(['familia', 'cantidad']).agg(
        f1_media=('f1_macro', 'mean'), f1_desv=('f1_macro', 'std'),
        f1_train=('f1_train', 'mean'), pliegues=('pliegue', 'nunique')).reset_index()
    resumen.to_csv(carpeta / 'curvas_resumen.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, familia in zip(axes, ['Bagging', 'AdaBoost']):
        r = resumen[resumen.familia == familia]
        ax.plot(r.cantidad, r.f1_media, marker='o', label='Validación')
        ax.fill_between(r.cantidad, r.f1_media-r.f1_desv.fillna(0),
                        r.f1_media+r.f1_desv.fillna(0), alpha=.15)
        ax.plot(r.cantidad, r.f1_train, '--', label='Entrenamiento')
        ax.set(xlabel='Árboles / rondas efectivas', ylabel='F1 macro', title=familia)
        ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / 'curvas_sesgo_varianza.png', dpi=150)
    plt.close(fig)
    joblib.dump({'firma': clave, 'tabla': tabla}, archivo)
    return tabla


def _fila(familia, fold, n, efectivos, yv, pv, yt, pt, tiempo):
    return {'familia': familia, 'pliegue': fold, 'cantidad': n,
            'estimadores_efectivos': efectivos,
            'f1_macro': f1_score(yv, pv, average='macro', zero_division=0),
            'f1_train': f1_score(yt, pt, average='macro', zero_division=0),
            'sensibilidad_enrolled': recall_score(yv, pv, labels=['enrolled'],
                                                  average='macro', zero_division=0),
            'entrenamiento_s': tiempo}


def correlacion_errores(resultados, nombres, y, carpeta):
    errores = pd.DataFrame({n: (resultados[n]['oof'] != np.asarray(y)).astype(int)
                             for n in nombres})
    matriz = errores.corr()
    matriz.to_csv(Path(carpeta) / 'correlacion_errores.csv')
    errores.to_csv(Path(carpeta) / 'indicadores_errores_oof.csv', index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matriz, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(nombres)), nombres, rotation=40, ha='right')
    ax.set_yticks(range(len(nombres)), nombres)
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            ax.text(j, i, f'{matriz.iloc[i,j]:.2f}', ha='center', va='center')
    fig.colorbar(im, ax=ax, label='Correlación de indicadores de error OOF')
    fig.tight_layout()
    fig.savefig(Path(carpeta) / 'correlacion_errores.png', dpi=150)
    plt.close(fig)
    return matriz
