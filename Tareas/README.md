# Tareas

## Tarea 02: preprocesamiento de Breast Cancer Wisconsin

En [tarea_02_preprocesamiento_breast_cancer_wisconsin.ipynb](tarea_02_preprocesamiento_breast_cancer_wisconsin.ipynb) reviso el conjunto original Breast Cancer Wisconsin y aplico tres técnicas: completar valores faltantes con la mediana, crear una variable a partir de dos mediciones y reducir las nueve mediciones a dos componentes con PCA. El notebook contiene las gráficas y los resultados de mi ejecución.

El archivo de datos no está incluido en este repositorio. Descarga `breast-cancer-wisconsin.data` desde la [página del conjunto original de UCI](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original). En Colab, ejecuta el notebook y sube el archivo cuando lo pida. Para usarlo localmente, guarda el archivo de datos en `Tareas/` y ejecuta el notebook desde un kernel de Python con `numpy`, `pandas`, `matplotlib`, `seaborn` y `scikit-learn` instalados.

## Tarea 03: regresión y clasificación con Wine Quality

En [tarea_03_wine_quality.ipynb](tarea_03_wine_quality.ipynb) uso los datos de vino tinto de [Wine Quality, de UCI](https://archive.ics.uci.edu/dataset/186/wine+quality). Primero intento predecir la calificación y después clasifico los vinos usando `quality >= 7`. En ambos casos utilizo regresión lineal y comparo el resultado con una referencia sencilla: el promedio o la clase mayoritaria.

El CSV se descarga al ejecutar el notebook, por lo que se necesita internet. En Colab, abre o sube el archivo y ejecuta las celdas en orden. Para usarlo localmente se necesitan Python, Jupyter, `numpy`, `pandas`, `matplotlib` y `scikit-learn`.
