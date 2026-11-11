"""Evaluación reproducible; ninguna predicción OOF procede de su entrenamiento."""

from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import hashlib
import json
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, recall_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from threadpoolctl import threadpool_limits


def slug(nombre):
    import unicodedata
    normal = unicodedata.normalize('NFKD', nombre).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]+', '_', normal.lower()).strip('_')


def jsonable(valor):
    if isinstance(valor, dict):
        return {str(k): jsonable(v) for k, v in valor.items()}
    if isinstance(valor, (tuple, list)):
        return [jsonable(v) for v in valor]
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, (str, int, float, bool)) or valor is None:
        return valor
    if hasattr(valor, 'get_params'):
        return {'clase': f'{type(valor).__module__}.{type(valor).__name__}',
                'parametros': jsonable(valor.get_params(deep=False))}
    if callable(valor) and hasattr(valor, '__qualname__'):
        return f'{valor.__module__}.{valor.__qualname__}'
    return repr(valor)


def guardar_json(ruta, valor):
    Path(ruta).write_text(json.dumps(jsonable(valor), ensure_ascii=False, indent=2),
                          encoding='utf-8')


def firma(*objetos):
    # Invalidar la caché si cambian datos, código, parámetros, CV o bibliotecas.
    codigo = [(p.name, hashlib.sha256(p.read_bytes()).hexdigest())
              for p in sorted(Path(__file__).parent.glob('*.py'))]
    versiones = {p: version(p) for p in ['scikit-learn', 'numpy', 'pandas',
                                        'lightgbm', 'catboost']}
    return joblib.hash((objetos, codigo, versiones))


def evaluar(nombre, modelo, X, y, cv, positiva, carpeta, cache, n_jobs=8,
            reutilizar=True, busqueda_s=0.0):
    """El F1 siempre sale de cross_validate(..., scoring='f1_macro')."""
    carpeta, cache = Path(carpeta), Path(cache)
    carpeta.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    cv_lista = list(cv.split(X, y)) if hasattr(cv, 'split') else list(cv)
    clave = firma(nombre, modelo, X, y, cv_lista, positiva, busqueda_s)
    archivo = cache / f'evaluacion_{slug(nombre)}.joblib'
    completos = all((carpeta / f'{prefijo}_{slug(nombre)}.{extension}').exists()
                    for prefijo, extension in [('cv', 'csv'), ('oof', 'csv'), ('resumen', 'json')])
    if reutilizar and archivo.exists() and completos:
        anterior = joblib.load(archivo)
        if anterior['firma'] == clave:
            print(f'{nombre}: resultados comprobados de caché', flush=True)
            return anterior['resultado']
    print(f'Evaluando {nombre} ({len(cv_lista)} pliegues)...', flush=True)
    with threadpool_limits(limits=1), parallel_backend('threading'):
        resultado = cross_validate(
            modelo, X, y, cv=cv if hasattr(cv, 'split') else cv_lista,
            scoring='f1_macro', n_jobs=n_jobs,
            return_estimator=True, return_indices=True, error_score='raise')
    pred = np.empty(len(y), dtype=np.asarray(y).dtype)
    cobertura = np.zeros(len(y), dtype=int)
    filas = []
    for i, (estimador, idx) in enumerate(zip(resultado['estimator'],
                                            resultado['indices']['test'])):
        with threadpool_limits(limits=1):
            pred_i = np.asarray(estimador.predict(X.iloc[idx])).ravel()
        pred[idx] = pred_i
        cobertura[idx] += 1
        filas.append({'pliegue': i + 1, 'f1_macro': resultado['test_score'][i],
                      'sensibilidad': recall_score(y.iloc[idx], pred_i,
                          labels=[positiva], average='macro', zero_division=0),
                      'entrenamiento_s': resultado['fit_time'][i],
                      'evaluacion_s': resultado['score_time'][i]})
    assert np.all(cobertura == 1), 'Cada fila debe predecirse una vez fuera de ajuste.'
    pliegues = pd.DataFrame(filas)
    resumen = {'modelo': nombre,
               'f1_media': pliegues.f1_macro.mean(),
               'f1_desv': pliegues.f1_macro.std(ddof=1),
               'sensibilidad_media': pliegues.sensibilidad.mean(),
               'sensibilidad_desv': pliegues.sensibilidad.std(ddof=1),
               'entrenamiento_media_s': pliegues.entrenamiento_s.mean(),
               'entrenamiento_total_s': pliegues.entrenamiento_s.sum(),
               'busqueda_s': busqueda_s}
    salida = {'resumen': resumen, 'pliegues': pliegues, 'oof': pred}
    pliegues.to_csv(carpeta / f'cv_{slug(nombre)}.csv', index=False)
    pd.DataFrame({'fila': np.arange(len(y)), 'real': y, 'prediccion': pred}).to_csv(
        carpeta / f'oof_{slug(nombre)}.csv', index=False)
    guardar_json(carpeta / f'resumen_{slug(nombre)}.json', resumen)
    joblib.dump({'firma': clave, 'resultado': salida}, archivo)
    print(f'{nombre}: F1={resumen["f1_media"]:.4f} ± {resumen["f1_desv"]:.4f}',
          flush=True)
    return salida


def optimizar(nombre, estimador, espacio, X, y, cv, carpeta, cache,
              n_iter=50, n_jobs=8, reutilizar=True):
    """Selecciona 50 candidatos distintos; evaluación posterior declara selección."""
    if n_iter < 50:
        raise ValueError('La entrega exige al menos 50 configuraciones por familia.')
    carpeta, cache = Path(carpeta), Path(cache)
    carpeta.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    clave = firma(nombre, estimador, espacio, X, y, cv, n_iter)
    archivo = cache / f'busqueda_{slug(nombre)}.joblib'
    completos = all((carpeta / f'{prefijo}_{slug(nombre)}.{extension}').exists()
                    for prefijo, extension in [('busqueda', 'csv'), ('mejor', 'json'), ('meseta', 'png')])
    if reutilizar and archivo.exists() and completos:
        anterior = joblib.load(archivo)
        if anterior['firma'] == clave:
            print(f'{nombre}: búsqueda comprobada de caché', flush=True)
            return anterior['estimador'], anterior['parametros'], anterior['segundos']
    print(f'Buscando {nombre}: {n_iter} configuraciones × 5 pliegues...', flush=True)
    busqueda = RandomizedSearchCV(
        estimador, espacio, n_iter=n_iter, scoring='f1_macro', cv=cv,
        n_jobs=n_jobs, random_state=2026, error_score='raise', refit=True,
        return_train_score=False, pre_dispatch=n_jobs)
    inicio = perf_counter()
    with threadpool_limits(limits=1), parallel_backend('threading'):
        busqueda.fit(X, y)
    segundos = perf_counter() - inicio
    candidatos = busqueda.cv_results_['params']
    unicos = {json.dumps(jsonable(p), sort_keys=True) for p in candidatos}
    assert len(unicos) >= 50, 'Se requieren 50 configuraciones efectivamente distintas.'
    assert np.isfinite(busqueda.cv_results_['mean_test_score']).all()
    tabla = pd.DataFrame({
        'configuracion': np.arange(1, len(candidatos) + 1),
        'f1_media': busqueda.cv_results_['mean_test_score'],
        'f1_desv_cv_poblacional': busqueda.cv_results_['std_test_score'],
        'entrenamiento_media_s': busqueda.cv_results_['mean_fit_time'],
        'parametros': [json.dumps(jsonable(p), ensure_ascii=False, sort_keys=True)
                       for p in candidatos]})
    for i in range(5):
        tabla[f'f1_pliegue_{i+1}'] = busqueda.cv_results_[f'split{i}_test_score']
    tabla['mejor_acumulado'] = tabla.f1_media.cummax()
    tabla.to_csv(carpeta / f'busqueda_{slug(nombre)}.csv', index=False)
    ganancia = tabla.mejor_acumulado.iloc[-1] - tabla.mejor_acumulado.iloc[-21]
    detalle = {'parametros': busqueda.best_params_, 'configuraciones_distintas': len(unicos),
               'mejor_f1_seleccion': busqueda.best_score_, 'busqueda_s': segundos,
               'ganancia_ultimas_20': ganancia, 'meseta_local_001': ganancia <= .001}
    guardar_json(carpeta / f'mejor_{slug(nombre)}.json', detalle)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.scatter(tabla.configuracion, tabla.f1_media, s=12, alpha=.5, label='Candidato')
    ax.plot(tabla.configuracion, tabla.mejor_acumulado, color='#114b5f',
            label='Mejor acumulado')
    ax.set(xlabel='Configuraciones evaluadas (orden de muestreo)', ylabel='F1 macro CV',
           title=f'{nombre}: ganancia final de 20 candidatos = {ganancia:.4f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / f'meseta_{slug(nombre)}.png', dpi=150)
    plt.close(fig)
    joblib.dump({'firma': clave, 'estimador': busqueda.best_estimator_,
                 'parametros': busqueda.best_params_, 'segundos': segundos}, archivo)
    print(f'{nombre}: selección F1={busqueda.best_score_:.4f}, {segundos:.1f}s', flush=True)
    return busqueda.best_estimator_, busqueda.best_params_, segundos


def tabla_resultados(resultados, carpeta, parametros=None):
    tabla = pd.DataFrame([v['resumen'] for v in resultados.values()]).sort_values(
        'f1_media', ascending=False, kind='stable').reset_index(drop=True)
    tabla['F1 macro (media ± desv)'] = tabla.apply(
        lambda f: f'{f.f1_media:.4f} ± {f.f1_desv:.4f}', axis=1)
    tabla['mejores_parametros'] = tabla.modelo.map(
        lambda n: json.dumps(jsonable((parametros or {}).get(n, {})), ensure_ascii=False))
    tabla.to_csv(Path(carpeta) / 'comparacion_modelos.csv', index=False)
    return tabla


def matriz_ganador(tabla, resultados, y, etiquetas, carpeta):
    nombre = tabla.iloc[0].modelo
    cm = confusion_matrix(y, resultados[nombre]['oof'], labels=etiquetas)
    pd.DataFrame(cm, index=etiquetas, columns=etiquetas).to_csv(
        Path(carpeta) / 'matriz_confusion_mejor.csv')
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=etiquetas).plot(ax=ax, cmap='Blues',
                                                            colorbar=False)
    ax.set(title=nombre, xlabel='Predicción', ylabel='Clase real')
    fig.tight_layout()
    fig.savefig(Path(carpeta) / 'matriz_confusion_mejor.png', dpi=150)
    plt.close(fig)
    return nombre, cm
