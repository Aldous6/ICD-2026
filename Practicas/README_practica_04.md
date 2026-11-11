Práctica 04: meta aprendizaje

El notebook [practica_04_metaaprendizaje.ipynb](practica_04_metaaprendizaje.ipynb), de Aldo Santiago Márquez, compara ensambles con los cinco clasificadores ya seleccionados en la Práctica 3 y busca un modelo para Bank Marketing. El código reutilizable está en [metaaprendizaje/](metaaprendizaje/). No se vuelve a optimizar árbol, Naive Bayes, kNN, SVM ni regresión logística sobre deserción.

Para deserción se conservan los CSV del curso en [datos/](datos/), los parámetros finales de [mejores_parametros.json](resultados_practica_03/mejores_parametros.json) y la asignación de estudiantes de [pliegues.csv](resultados_practica_03/pliegues.csv). La procedencia y las huellas de los datos originales están en [datos/README.md](datos/README.md). El conjunto de entrenamiento tiene 3539 estudiantes; se usan los 30 atributos disponibles en el archivo de prueba, correspondientes al ingreso y primer semestre. Se excluyen `id`, `Target` y los atributos del segundo semestre. Las etiquetas son `dropout`, `enrolled` y `graduate`. La división de prueba de 885 estudiantes no tiene etiquetas y no permite medir F1.

La fuente original, [Predict Students' Dropout and Academic Success, UCI](https://archive.ics.uci.edu/dataset/697), describe la situación al final de la duración normal de la carrera. Se respeta la división proporcionada por el curso en vez de reconstruirla con los 4424 registros públicos. La evaluación reutiliza los mismos cinco pliegues estratificados de la Práctica 3, con semilla 42, y verifica la correspondencia por identificador.

Para Bank se utiliza exclusivamente `datos/bank-additional-full.csv`, con separador `;`: 41188 filas y 20 atributos antes de retirar `duration`. No se utiliza `bank-full.csv`, que pertenece a otra versión. La fuente es [Bank Marketing, UCI](https://archive.ics.uci.edu/dataset/222), publicada por Moro, Rita y Cortez, DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306), bajo licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). El objetivo indica si se contrató un depósito a plazo. Se codifica `yes` como 1 y `no` como 0; `X` conserva exactamente las columnas originales salvo `duration` y `y`. Las transformaciones adicionales se realizan dentro del modelo. `duration` se excluye porque describe la duración de una llamada ya realizada y no está disponible para decidir a quién llamar antes del contacto.

La ficha de UCI indica que Bank está ordenado cronológicamente entre mayo de 2008 y noviembre de 2010. La consigna requiere `StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)`, que mezcla fechas. Se utiliza ese protocolo para la comparación del curso; sus resultados no estiman por sí solos la capacidad de predecir campañas futuras. Tampoco permiten atribuir causas al éxito de una llamada. Ambos conjuntos proceden de instituciones concretas y no se supone que representen a cualquier institución mexicana.

Cada preprocesamiento se ajusta en el entrenamiento de su pliegue mediante un `Pipeline`. Se conservan la codificación de categorías, discretización, escalado y atributos derivados de los cinco modelos finales de la Práctica 3. La SVM usada en votación suave y stacking se calibra con [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html), envolviendo su Pipeline completo; la calibración permanece dentro de cada partición de entrenamiento. Los pesos de clase se adaptan a la codificación interna de etiquetas de los ensambles sin cambiar su valor relativo.

La comparación de varianza enfrenta un árbol sin límite de profundidad con bagging hasta 400 árboles; la curva utiliza las cantidades 1, 5, 10, 20, 50, 100, 200, 300 y 400. [warm_start](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) incorpora árboles progresivamente dentro de cada pliegue. La comparación de sesgo enfrenta un tocón de profundidad 1 con AdaBoost hasta 400 rondas y obtiene las predicciones intermedias mediante [staged_predict](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html). El árbol y el tocón independientes se evalúan aparte. La explicación sobre varianza y sesgo es conceptual: una curva de F1 no mide directamente la descomposición estadística de ambos términos.

La votación dura, la votación suave y el [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) combinan los cinco modelos base. El meta-modelo es logístico y se entrena con predicciones internas fuera de muestra. La evaluación externa del stacking usa `cross_validate` con los cinco pliegues comunes; sus pliegues internos no sustituyen esa evaluación. La correlación de errores se calcula entre indicadores de clasificación incorrecta de las predicciones fuera de muestra, alineadas por estudiante.

Random Forest, Extra-Trees, LightGBM y CatBoost prueban 50 configuraciones distintas por familia en cada uno de los dos conjuntos mediante [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html). Los espacios son finitos y se muestrean sin reemplazo. Se registran los candidatos, parámetros, tiempos y la evolución del mejor F1 acumulado en el orden de muestreo; una ganancia de hasta 0.001 en los últimos 20 candidatos se identifica como meseta local. Los espacios incluyen selección de atributos por árbol, complejidad de hojas o profundidad, regularización y pesos de clase; Extra-Trees también compara bootstrap. Las bibliotecas de boosting exploran tasa de aprendizaje y submuestreo de filas y columnas. En [LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html) se activa la frecuencia de submuestreo; en [CatBoost](https://catboost.ai/docs/en/references/training-parameters/common) se emplean opciones de bootstrap compatibles con `subsample`. Una meseta local no demuestra que se alcanzó el óptimo global.

Las búsquedas y la evaluación de sus configuraciones ganadoras reutilizan los mismos cinco pliegues. Esta es una comparación de selección, no una validación anidada ni una prueba independiente: el mejor resultado puede ser optimista por haber elegido sus parámetros con esos datos. Lo mismo ocurre con las configuraciones finales de la Práctica 3, seleccionadas previamente sobre este entrenamiento. Sus nuevos F1 no se presentan como una reproducción de los F1 anidados originales. `best_score_` se conserva como resultado de selección y no como una segunda evaluación independiente. La elección entre familias también utiliza la validación; pequeñas diferencias no prueban superioridad estadística.

En Bank cada F1 de la tabla se obtiene mediante `cross_validate(modelo, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=2026), scoring="f1_macro")`. Se incluyen referencias Dummy, árbol, Naive Bayes, kNN y logística con y sin balanceo, además de las cuatro familias de ensambles y una logística optimizada con 50 configuraciones. Las búsquedas de los cuatro ensambles comparan one-hot con codificación ordinal; la de logística compara regularización, pesos y agrupación de categorías poco frecuentes.

También se evalúa `CatBoost categórico`, conservando los hiperparámetros del clasificador seleccionados en su búsqueda de 50 configuraciones y cambiando el tratamiento de las categorías. Un `ColumnTransformer` imputa dentro del Pipeline y devuelve un DataFrame con las categorías de texto; `cat_features` las identifica para su [tratamiento nativo en CatBoost](https://catboost.ai/docs/en/features/categorical-features). Se evalúa esta alternativa con los mismos cinco pliegues, sin ejecutar otra búsqueda de hiperparámetros. Es una comparación de representación con parámetros fijos, no una optimización específica de la variante nativa.

Votación suave y stacking combinan las tres mejores familias simples distintas de Bank entre las cuatro optimizadas y logística. Para CatBoost se elige su mejor representación codificada o nativa y se cuenta como una sola familia. Se evalúan además siete alternativas con [TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html): Random Forest, Extra-Trees, LightGBM, logística optimizada, las dos representaciones de CatBoost y la configuración transferida. El ajuste de umbral se prueba en todas estas alternativas, aunque no integren las tres familias seleccionadas para combinar. El optimizador envuelve el Pipeline completo y busca el umbral mediante tres pliegues internos únicamente dentro del entrenamiento externo. La elección de familias, representación y procedimiento usa la tabla de validación y puede introducir optimismo.

Se vuelve a entrenar la familia ganadora de deserción con los atributos de Bank: no se aplican sus coeficientes ni árboles ya entrenados a un problema con otro esquema y objetivo. Los pesos ligados a los nombres de las tres clases académicas se retiran al transferir; los pesos automáticos se recalculan con las clases bancarias.

Las tablas ordenan los modelos por la media del F1 macro de los cinco pliegues y muestran su desviación estándar muestral (`ddof=1`), sensibilidad de `enrolled` o `yes` y tiempo de entrenamiento. `entrenamiento_media_s` es la media por pliegue de `fit_time`; los CSV incluyen también `entrenamiento_total_s`, la suma de los cinco ajustes. En stacking y los modelos con umbral ese tiempo incluye sus ajustes internos. `busqueda_s` informa por separado el tiempo total de la búsqueda y su reajuste final; vale cero cuando esa alternativa no ejecuta una búsqueda adicional, aunque reutilice parámetros de otra búsqueda. Las desviaciones no son intervalos de confianza. Las matrices de confusión se muestran únicamente para el ganador de cada conjunto y usan predicciones fuera de muestra, nunca predicciones sobre su propio entrenamiento. El notebook genera las comparaciones y conclusiones a partir de los resultados calculados.

En la ejecución guardada del 7 de noviembre de 2026, [LightGBM ganó en deserción](resultados_practica_04/desercion/comparacion_modelos.csv) con F1 macro **0.6968 ± 0.0139**, sensibilidad de `enrolled` **0.5638** y entrenamiento medio por pliegue de **0.76 s**. La mejor base fija fue SVM, con F1 **0.6918**: la diferencia fue **+0.49 puntos porcentuales**. En [Bank ganó LightGBM transferido con umbral interno](resultados_practica_04/bank/comparacion_modelos.csv), con F1 **0.7180 ± 0.0077**, sensibilidad de `yes` **0.5532** y entrenamiento medio de **14.31 s**. Coincidió la familia ganadora; transferirla sin ajustar el umbral obtuvo **0.6959**. El modelo final se guardó en [Aldo_Santiago_Marquez-umbral_interno_transferido_lightgbm.joblib](Aldo_Santiago_Marquez-umbral_interno_transferido_lightgbm.joblib), con umbral **0.6126179555297282** tras reajustar con todos los datos. Estas cifras reutilizan los pliegues de selección, por lo que conservan el optimismo ya descrito; cuatro búsquedas no alcanzaron el criterio local de meseta. La [discusión completa](resultados_practica_04/conclusiones.md) compara las familias y las limitaciones.

Se recomienda Python 3.12. Desde la raíz del repositorio, instala las versiones declaradas y abre el notebook:

```powershell
python -m pip install -r Practicas/requirements_practica_04.txt
jupyter lab Practicas/practica_04_metaaprendizaje.ipynb
```

En Jupyter selecciona **Kernel → Restart Kernel and Run All Cells**. Las rutas son relativas; abre el notebook desde la raíz del repositorio o desde `Practicas/`. Conserva junto al notebook el paquete `metaaprendizaje`, los datos y `resultados_practica_03`. Las búsquedas completas requieren tiempo y no se conoce su duración antes de ejecutarlas en el equipo. No es necesario ejecutar `flutter run` ni `npm run`.

También se puede ejecutar el notebook completo desde la terminal. El script utiliza un kernel con el mismo intérprete que lo invoca y guarda las salidas al completar cada celda; si una celda falla, conserva las salidas anteriores y propaga el error:

```powershell
python Practicas/ejecutar_practica_04.py
```

Las nueve comprobaciones funcionales de la entrega se ejecutan con el comando siguiente y pasaron en el entorno declarado. No sustituyen la ejecución completa del notebook:

```powershell
python Practicas/tests_practica_04.py
```

`REUTILIZAR_CACHE=True` recupera los resultados de `Practicas/.cache_practica_04/` únicamente si coinciden datos, modelos, particiones, código y versiones de las bibliotecas verificadas. No se carga una tabla ajena ni se sustituyen experimentos por resultados precalculados de otra configuración. Para volver a medir todos los ajustes desde cero, cambia a `REUTILIZAR_CACHE=False` y ejecuta Restart & Run All. Los tiempos recuperados de caché corresponden a su cálculo original; no miden el tiempo de lectura de caché.

Los resultados se escriben en `resultados_practica_04/desercion/` y `resultados_practica_04/bank/`. Cada carpeta contiene `comparacion_modelos.csv`, resultados por pliegue y predicciones OOF por modelo, búsquedas, mejores configuraciones y sus gráficas de meseta. Deserción también guarda curvas de sesgo/varianza y correlación de errores. `resultados_practica_04/conclusiones.md` y `metadatos.json` registran la comparación, semillas, versiones y huellas. Guarda el notebook con sus salidas después de la ejecución completa. El modelo ganador de Bank se reajusta con todos sus datos y se guarda directamente como estimador en `Aldo_Santiago_Marquez-nombre_modelo.joblib`, sustituyendo `nombre_modelo` por el nombre real normalizado del ganador. No es un diccionario de metadatos. La ejecución comprueba que el modelo recargado conserva sus predicciones. Para entregar, se necesitan el enlace al repositorio, el notebook ejecutado y ese archivo `.joblib`. La plataforma del curso se entrega manualmente.

Para cargar el modelo en otro proceso deben estar disponibles las mismas dependencias y el paquete `metaaprendizaje` en la ruta de importación de Python. Desde la raíz del repositorio, este ejemplo obtiene el nombre exacto del archivo de los metadatos de la ejecución:

```python
from pathlib import Path
import json
import sys
import joblib
import pandas as pd

sys.path.insert(0, "Practicas")
base = Path("Practicas")
metadata = json.loads((base / "resultados_practica_04/metadatos.json").read_text(encoding="utf-8"))
modelo = joblib.load(base / metadata["modelo_entrega"])
bank = pd.read_csv(base / "datos/bank-additional-full.csv", sep=";")
predicciones = modelo.predict(bank.drop(columns=["duration", "y"]))
```

El archivo incluye el preprocesamiento y recibe un DataFrame con los 19 atributos originales de Bank, sin `duration` ni `y`. La instalación, el código y la evaluación reproducible forman parte de la entrega; las métricas no se pueden conocer sin ejecutar los experimentos.
