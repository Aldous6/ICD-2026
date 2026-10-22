**Python Data Science Handbook: revisión por bloques**

La entrega consta de cinco notebooks nuevos con explicaciones en español, ejemplos ejecutables, ejercicios resueltos y conclusiones. Cada archivo funciona por separado y contiene las fuentes de las secciones del libro que aborda. Las salidas y las gráficas están guardadas para revisar el trabajo directamente en GitHub.

El [bloque 01: IPython y Jupyter](bloque_01_ipython_jupyter.ipynb) practica ayuda y documentación, comandos mágicos, historial, interacción con el sistema, ejecución de scripts, errores y medición de tiempos.

El [bloque 02: NumPy](bloque_02_numpy.ipynb) trabaja tipos, dimensiones, vistas y copias, funciones universales, agregaciones, broadcasting, máscaras, índices avanzados, ordenamiento y arreglos estructurados con mediciones sintéticas.

El [bloque 03: pandas](bloque_03_pandas.ipynb) transforma un CSV sintético con inconsistencias en un resumen de ventas estimadas. Incluye índices, alineación, faltantes, cadenas, concatenación, uniones, agrupación, pivotes, índices jerárquicos y series de tiempo. Las imputaciones se identifican para distinguir observaciones de estimaciones.

El [bloque 04: Matplotlib](bloque_04_matplotlib.ipynb) incluye líneas, dispersión, barras de error, histogramas, contornos, imágenes, leyendas, barras de color, paneles, anotaciones, estilos, superficies 3D, proyecciones y seaborn. El ejemplo de proyección calcula una cuadrícula Mercator sin instalar Basemap.

El [bloque 05: aprendizaje automático](bloque_05_aprendizaje_automatico.ipynb) revisa clasificación, regresión, ingeniería de características, validación cruzada, búsqueda de hiperparámetros, curvas de aprendizaje, Naive Bayes, SVM, árboles, bosques, regularización, PCA, Isomap, k-means, mezclas gaussianas, KDE y una aplicación con imágenes de dígitos. El escalado supervisado se ajusta dentro de pipelines y la evaluación final usa datos reservados.

**Uso y dependencias**

Se verificaron los cinco notebooks con Python 3.12.10 y las versiones indicadas en [requirements.txt](requirements.txt). Se validó su estructura y se ejecutaron sus 60 celdas de código desde kernels nuevos, sin errores; se guardaron 17 gráficas. El error capturado del bloque 01 es un ejemplo deliberado de diagnóstico y no interrumpe la ejecución. Las mediciones de tiempo dependen del equipo.

Para ejecutarlos localmente, desde la raíz del repositorio instala las dependencias con `python -m pip install -r Notebooks/requirements.txt` y abre `jupyter lab Notebooks/`. Selecciona un kernel de ese entorno y ejecuta las celdas en orden. No se necesitan archivos de datos adicionales ni descargas durante la ejecución: los datos son sintéticos o se incluyen con scikit-learn. La instalación inicial de paquetes sí requiere acceso a ellos.

En Google Colab se puede subir cualquiera de los cinco archivos y ejecutar todas sus celdas. Si falta una biblioteca, instálala en ese entorno antes de continuar. Las versiones del entorno de Colab pueden diferir de las verificadas localmente.

**Procedencia**

Referencia de la consigna: VanderPlas, J. (2022). *Python Data Science Handbook: Essential Tools for Working with Data*. O’Reilly Media.

Se revisó el [índice de notebooks_v1](https://github.com/husseinlopez/PythonDataScienceHandbook/blob/master/notebooks_v1/Index.ipynb) del repositorio indicado. Sus cinco capítulos contienen 65 notebooks, incluidas introducciones y recursos. El [prefacio de esos materiales](https://github.com/husseinlopez/PythonDataScienceHandbook/blob/master/notebooks_v1/00.00-Preface.ipynb) identifica copyright de 2016. Por ello, estos cuadernos documentan el material `notebooks_v1` solicitado, sin atribuirlo como contenido verificado de la edición de 2022.

Cada notebook nuevo enlaza las secciones correspondientes y la documentación de las bibliotecas consultada. Los ejemplos son una síntesis práctica de los temas; las explicaciones y ejercicios son propios. Los dos extractos originales que también están en esta carpeta son material de consulta del libro. Sus enlaces e imágenes locales dependen de otros archivos del repositorio fuente, accesibles desde el índice enlazado.

El [repositorio fuente](https://github.com/husseinlopez/PythonDataScienceHandbook) distingue texto con licencia CC BY-NC-ND y código con licencia MIT. Las fuentes de Iris y dígitos se documentan mediante los cargadores oficiales de scikit-learn dentro del bloque 05.
