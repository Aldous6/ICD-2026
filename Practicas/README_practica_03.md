# Práctica 03: clasificación de deserción estudiantil

El notebook [practica_03_clasificacion_desercion.ipynb](practica_03_clasificacion_desercion.ipynb) compara árbol de decisión, Naive Bayes, kNN, SVM y regresión logística para predecir, al cierre del primer semestre, si un estudiante terminará como desertor (`dropout`), inscrito (`enrolled`) o graduado (`graduate`) al final de la duración normal de su carrera. No utiliza ensambles ni redes neuronales.

## Dataset y archivos de la entrega

La fuente es [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697), de UCI, con 4424 estudiantes y 36 atributos. La consigna del curso establece una división propia: `desercion-entrenamiento.csv`, con 3539 estudiantes y `Target`, y `desercion-prueba.csv`, con 885 estudiantes sin objetivo y únicamente los atributos disponibles al cierre del primer semestre. Los códigos numéricos de variables nominales se interpretan como categorías. La procedencia y el estado de los archivos se documentan en [datos/README.md](datos/README.md).

Los dos CSV originales están incluidos en `Practicas/datos/`. Se descargaron del [repositorio del curso](https://github.com/husseinlopez/icd2026/tree/main/datos) el 4 de noviembre de 2026, sin modificar sus bytes. El entrenamiento contiene 1767 graduados, 1137 desertores y 635 inscritos; ninguno de los archivos tiene valores faltantes. Sus identificadores son únicos y no se comparten entre entrenamiento y prueba. La ejecución refinada completa del 4 de noviembre de 2026 conserva las salidas del notebook, la comparación de ambos semestres, los metadatos y el [CSV de entrega](predicciones_desercion.csv) regenerado. La [referencia inicial](resultados_practica_03/referencia_inicial/) conserva los resultados y las predicciones anteriores. No se reconstruye la división a partir del conjunto público de UCI.

## Método y reproducibilidad

Todos los clasificadores llevan su preprocesamiento dentro de un `Pipeline`, ajustado en cada partición de entrenamiento. La comparación utiliza validación cruzada estratificada anidada con cinco pliegues externos y tres internos; los cinco modelos comparten las mismas particiones. Árbol, Naive Bayes y kNN evalúan una fase original de 80–160 candidatos. SVM y regresión logística comparan una fase original con otra refinada dentro de cada búsqueda interna: 80–160 candidatos por fase, 160–320 en total. Se selecciona el mejor candidato por F1 interno entre ambas fases, sin combinar predicciones.

La fase refinada compara atributos originales con tasas académicas derivadas, como aprobadas/inscritas, y otros indicadores de avance. También busca la agrupación opcional de categorías poco frecuentes y el peso relativo del grupo one-hot, además de C, gamma y los pesos de clase según corresponda. Las transformaciones están dentro del Pipeline; los agrupamientos y escaladores se aprenden en cada ajuste. Las características derivadas usan únicamente los atributos disponibles al decidir y no consultan `Target`. En el modelo de entrega se utiliza solo el primer semestre.

Cada fase puede ampliarse en bloques de 40 candidatos y tiene su propio criterio de meseta: mejora del mejor F1 macro acumulado menor o igual a 0.001 en las últimas 20 configuraciones. Solo se declara meseta conjunta cuando todas las fases cumplen esa regla. Después se optimiza cada familia sobre todo el entrenamiento, con los mismos cinco pliegues: en SVM y logística la fase refinada final evalúa al menos 120 candidatos, por lo que el presupuesto final es 200–320; las otras tres familias mantienen 80–160. Este presupuesto adicional es selección final y no cambia la estimación externa.

En los resultados actuales del primer semestre, las búsquedas finales de SVM y logística evaluaron 200 candidatos cada una (80 originales y 120 refinados); las otras tres evaluaron 80. Todas cumplieron su criterio de meseta. Las búsquedas internas sumaron 800 candidatos para SVM, 880 para logística y 400 para cada otra familia. Se registran candidatos, fases, parámetros, métricas y tiempos. Las semillas son 42 para los pliegues y la fase original y 2026 para el muestreo refinado; se usaron ocho procesos paralelos y el notebook declara las versiones instaladas. Una meseta local no demuestra un óptimo global.

El notebook enlaza la documentación oficial de [validación anidada](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) y [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html), entre otras referencias que sustentan el procedimiento.

La evaluación incluye F1 macro, sensibilidad y especificidad por clase, y una matriz de confusión construida con predicciones externas a cada entrenamiento. La comparación entre uno y dos semestres vuelve a optimizar la familia ganadora y utiliza los mismos pliegues para estimar la diferencia. El modelo destinado a prueba usa exclusivamente los atributos del primer semestre.

## Resultados de la ejecución guardada

El F1 macro es la media ± desviación estándar muestral de los cinco pliegues externos. Los tiempos son segundos medidos de búsqueda; la columna interna suma las cinco optimizaciones anidadas y la final corresponde a la búsqueda sobre todo el entrenamiento. No son tiempos de predicción ni una medición de toda la ejecución.

| Clasificador | F1 macro externo | Búsquedas internas (s) | Búsqueda final (s) |
| --- | --- | ---: | ---: |
| SVM | 0.6883 ± 0.0106 | 765.92 | 677.72 |
| Regresión logística | 0.6842 ± 0.0057 | 184.68 | 90.28 |
| Árbol | 0.6378 ± 0.0128 | 38.10 | 12.78 |
| Naive Bayes | 0.6104 ± 0.0118 | 27.00 | 9.42 |
| kNN | 0.5627 ± 0.0121 | 78.34 | 29.74 |

Se eligió SVM por su mayor F1 macro medio. Su ventaja sobre logística es pequeña: 0.0041, aproximadamente 0.41 puntos porcentuales. La logística requirió menos tiempo de búsqueda y obtuvo mayor sensibilidad media para desertores: 0.7009 frente a 0.6799 de SVM; SVM tuvo mayor especificidad para esa clase, 0.9338 frente a 0.9242. Las desviaciones entre pliegues no son intervalos de confianza ni una prueba de superioridad estadística.

La SVM inicial obtuvo 0.6784 ± 0.0138; la versión refinada mejora su media en 0.0099, aproximadamente 0.99 puntos porcentuales, sobre los mismos pliegues. La logística pasa de 0.6707 a 0.6842. La [comparación de versiones](resultados_practica_03/comparacion_versiones.csv) registra esos cambios; la [referencia inicial](resultados_practica_03/referencia_inicial/) conserva las tablas y predicciones originales con huellas SHA-256. El CSV refinado cambia 34 de las 885 predicciones respecto de esa referencia. La ampliación del diseño después de observar resultados puede introducir optimismo de desarrollo; la elección entre familias y la mejora aparente deben confirmarse en la prueba del curso. Sus etiquetas no se conocen y aumentar el F1 de validación no garantiza aumentar el F1 de prueba.

En las predicciones externas de SVM, 773 de 1137 desertores fueron identificados correctamente; 211 se confundieron con inscritos y 153 con graduados. Los desertores predichos como graduados aprobaron en promedio 5.824 unidades del primer semestre, frente a 1.620 de los correctamente identificados como desertores. Esto sugiere, como hipótesis, que un inicio académico favorable puede ocultar abandonos posteriores; los datos no establecen sus causas.

Al incorporar el segundo semestre y volver a optimizar SVM con el procedimiento refinado en los mismos pliegues, el F1 macro externo fue 0.7234 ± 0.0058. La diferencia pareada fue +0.0351 ± 0.0092, unos 3.51 puntos porcentuales; fue positiva en los cinco pliegues. La sensibilidad de desertores aumentó en promedio 0.0404. La mejora debe ponderarse frente al semestre que se retrasa el apoyo: el notebook propone actuar con la primera predicción y actualizarla después. No se conoce el F1 de los 885 estudiantes de prueba.

Las tablas, las curvas de meseta, la matriz de confusión y las conclusiones están en [resultados_practica_03/](resultados_practica_03/); [metadatos.json](resultados_practica_03/metadatos.json) registra versiones, huellas, fases, parámetros y configuraciones evaluadas.

## Instrucciones de uso

Se recomienda Python 3.12. Los dos CSV originales del curso están juntos en `Practicas/datos/`; el notebook también admite `datos/` desde la raíz del repositorio. Conserva los nombres indicados y el identificador de los estudiantes de prueba. Desde la raíz instala las dependencias y abre el notebook:

```bash
python -m pip install -r Practicas/requirements_practica_03.txt
jupyter lab Practicas/practica_03_clasificacion_desercion.ipynb
```

El notebook puede revisarse con sus salidas y gráficas guardadas sin recalcular las búsquedas. Para reproducirlo, selecciona **Kernel → Restart Kernel and Run All Cells**. Las rutas son relativas y la ejecución no necesita descargar el conjunto de UCI. Revisa las comprobaciones de los archivos, los resultados de las búsquedas y los avisos sobre convergencia o meseta antes de entregar. Las búsquedas anidadas requieren tiempo; los tiempos medidos se muestran arriba y pueden variar según el equipo.

Las salidas y los tiempos guardados se obtuvieron con ocho procesos de joblib (`loky`). La fuente usa ahora ocho hilos (`threading`), con BLAS limitado a un hilo por cálculo, para evitar mensajes de `resource_tracker` al cerrar procesos en Windows. Una comprobación puntual con la configuración final de cada una de las cinco familias y dos pliegues produjo exactamente los mismos scores y predicciones de reajuste con ambos backends. El alcance de esta comprobación y el backend de la ejecución guardada constan en los metadatos; los tiempos de una nueva ejecución pueden variar.

Con los datos incluidos, el propio notebook genera [Practicas/predicciones_desercion.csv](predicciones_desercion.csv), con exactamente las columnas `id` y `Target`, 885 filas y las etiquetas `dropout`, `enrolled` o `graduate`. El archivo vigente ya fue regenerado con el modelo refinado y conserva los identificadores y el orden del archivo de prueba. También guarda tablas, gráficas, el diccionario de atributos y metadatos de ejecución en `Practicas/resultados_practica_03/`. Si se usa `datos/` en la raíz, las salidas se guardan en esa misma raíz: siempre se toma como base el directorio que contiene la carpeta de datos.

Para entregar, guarda las salidas del notebook después de una ejecución completa y sube el enlace al repositorio, `Practicas/practica_03_clasificacion_desercion.ipynb` y `Practicas/predicciones_desercion.csv` a la plataforma del curso. Las métricas de prueba solo pueden calcularlas los profesores con las etiquetas que no se proporcionaron.
