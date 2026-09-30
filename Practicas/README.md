# Práctica 01: análisis exploratorio de inspecciones sanitarias

El notebook [practica_01_eda.ipynb](practica_01_eda.ipynb) analiza inspecciones de establecimientos de alimentos realizadas en Chicago durante 2024. Revisa estructura, valores faltantes, categorías inconsistentes y relaciones entre el resultado de la inspección, el riesgo y las infracciones.

## Dataset

Se usa [Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5), publicado por la Ciudad de Chicago. Los registros provienen de inspecciones del programa de protección de alimentos del Chicago Department of Public Health. Cada fila representa una inspección; un establecimiento puede aparecer varias veces. El notebook consulta la [API del dataset](https://dev.socrata.com/foundry/data.cityofchicago.org/4ijn-s7e5) y filtra las fechas del 1 de enero al 31 de diciembre de 2024.

La ejecución guardada en el notebook contiene 19,228 registros y 17 atributos, entre ellos `inspection_date`, `facility_type`, `risk`, `results`, `violations`, `latitude` y `longitude`. `results` se considera una posible variable objetivo para un análisis posterior. El conjunto presenta valores faltantes y variantes en la escritura de `city`. La cantidad de infracciones usada en las gráficas se **estima** contando los separadores `|` del texto en `violations`; los valores faltantes se conservan. Consulta las [condiciones de uso de los datos](https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html).

## Requisitos

Se necesita conexión a internet para descargar el CSV al ejecutar el notebook. En Google Colab se requiere un entorno de Python con `numpy`, `pandas`, `matplotlib` y `seaborn`. Para ejecutarlo en una computadora también se necesitan Python 3 y JupyterLab. El CSV se descarga directamente desde la API; no hay que preparar un archivo de datos en el repositorio.

## Instrucciones de uso

En Google Colab, abre o sube [el notebook](practica_01_eda.ipynb) y selecciona **Entorno de ejecución → Ejecutar todas**. Si falta alguna biblioteca, instálala en una celda con `%pip install numpy pandas matplotlib seaborn` y vuelve a ejecutar desde el principio.

Para usarlo localmente, desde la raíz del repositorio instala las dependencias y abre el notebook:

```bash
python -m pip install numpy pandas matplotlib seaborn jupyterlab
jupyter lab Practicas/practica_01_eda.ipynb
```

En JupyterLab, ejecuta todas las celdas en orden desde un kernel nuevo. La fuente se actualiza, por lo que una nueva descarga puede producir cifras distintas de las salidas guardadas. El análisis es descriptivo y no establece relaciones causales.

Para la entrega, comparte el enlace de este repositorio y sube `Practicas/practica_01_eda.ipynb` a la plataforma del curso.

# Práctica 02: preprocesamiento de Breast Cancer Wisconsin

En [practica_02_preprocesamiento_breast_cancer_wisconsin.ipynb](practica_02_preprocesamiento_breast_cancer_wisconsin.ipynb) reviso el conjunto original Breast Cancer Wisconsin y aplico tres técnicas: completar valores faltantes con la mediana, crear una variable a partir de dos mediciones y reducir las nueve mediciones a dos componentes con PCA. El notebook contiene las gráficas y los resultados de mi ejecución.

El archivo de datos no está incluido en este repositorio. Descarga `breast-cancer-wisconsin.data` desde la [página del conjunto original de UCI](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original). En Colab, ejecuta el notebook y sube el archivo cuando lo pida. Para usarlo localmente, guarda el archivo de datos en `Practicas/` y ejecuta el notebook desde un kernel de Python con `numpy`, `pandas`, `matplotlib`, `seaborn` y `scikit-learn` instalados.
