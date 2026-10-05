# Prácticas

La carpeta contiene la [práctica 01 de análisis exploratorio](practica_01_eda.ipynb) y la [práctica 02 de preprocesamiento de riesgo crediticio](practica_02_preprocesamiento_riesgo_crediticio.ipynb). Ambos notebooks incluyen comentarios, gráficas y resultados de ejecución.

## Práctica 01: análisis exploratorio de inspecciones sanitarias

El notebook [practica_01_eda.ipynb](practica_01_eda.ipynb) analiza inspecciones de establecimientos de alimentos realizadas en Chicago durante 2024. Revisa estructura, valores faltantes, categorías inconsistentes y relaciones entre el resultado de la inspección, el riesgo y las infracciones.

### Dataset

Se usa [Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5), publicado por la Ciudad de Chicago. Los registros provienen de inspecciones del programa de protección de alimentos del Chicago Department of Public Health. Cada fila representa una inspección; un establecimiento puede aparecer varias veces. El notebook consulta la [API del dataset](https://dev.socrata.com/foundry/data.cityofchicago.org/4ijn-s7e5) y filtra las fechas del 1 de enero al 31 de diciembre de 2024.

La ejecución guardada en el notebook contiene 19,228 registros y 17 atributos, entre ellos `inspection_date`, `facility_type`, `risk`, `results`, `violations`, `latitude` y `longitude`. `results` se considera una posible variable objetivo para un análisis posterior. El conjunto presenta valores faltantes y variantes en la escritura de `city`. La cantidad de infracciones usada en las gráficas se **estima** contando los separadores `|` del texto en `violations`; los valores faltantes se conservan. Consulta las [condiciones de uso de los datos](https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html).

### Requisitos

Se necesita conexión a internet para descargar el CSV al ejecutar el notebook. En Google Colab se requiere un entorno de Python con `numpy`, `pandas`, `matplotlib` y `seaborn`. Para ejecutarlo en una computadora también se necesitan Python 3 y JupyterLab. El CSV se descarga directamente desde la API; no hay que preparar un archivo de datos en el repositorio.

### Instrucciones de uso

En Google Colab, abre o sube [el notebook](practica_01_eda.ipynb) y selecciona **Entorno de ejecución → Ejecutar todas**. Si falta alguna biblioteca, instálala en una celda con `%pip install numpy pandas matplotlib seaborn` y vuelve a ejecutar desde el principio.

Para usarlo localmente, desde la raíz del repositorio instala las dependencias y abre el notebook:

```bash
python -m pip install numpy pandas matplotlib seaborn jupyterlab
jupyter lab Practicas/practica_01_eda.ipynb
```

En JupyterLab, ejecuta todas las celdas en orden desde un kernel nuevo. La fuente se actualiza, por lo que una nueva descarga puede producir cifras distintas de las salidas guardadas. El análisis es descriptivo y no establece relaciones causales.

Para la entrega, comparte el enlace de este repositorio y sube `Practicas/practica_01_eda.ipynb` a la plataforma del curso.

## Práctica 02: exploración y preprocesamiento de riesgo crediticio

El notebook [practica_02_preprocesamiento_riesgo_crediticio.ipynb](practica_02_preprocesamiento_riesgo_crediticio.ipynb) aborda la preparación de datos como un primer análisis del equipo de una startup que ofrece crédito. Incluye EDA y tres tipos de procesamiento: limpieza de categorías sin significado documentado, extracción de características del historial y reducción de 13 mediciones financieras con PCA. Cada técnica tiene una comparación gráfica antes y después, su justificación y sus límites. No se entrena un clasificador.

### Dataset y reproducibilidad

Se utiliza [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), de UCI: 30 000 clientes, 23 atributos explicativos, un ID y el indicador de impago. Los registros corresponden a Taiwán y contienen historial mensual de abril a septiembre de 2005; los importes están en NT$. Es un caso académico y no se considera representativo de una fintech mexicana actual.

La [copia original del CSV](datos/default_credit_card_clients.csv) está incluida en el repositorio. Su [documentación de procedencia](datos/README.md) registra la fuente, la licencia CC BY 4.0 y la huella SHA-256. El notebook comprueba esa huella, conserva los datos originales y usa una semilla fija. El escalado y PCA se ajustan solo con el 80 % de los clientes; la comparación gráfica utiliza los mismos clientes del 20 % restante. El EDA describe el conjunto completo y no se reporta rendimiento predictivo.

### Instrucciones de uso

En Colab, abre o sube el notebook y selecciona **Entorno de ejecución → Ejecutar todas**. Si solo subes el notebook, necesita internet para descargar el CSV de UCI. Para trabajar sin esa descarga, sube también `default_credit_card_clients.csv` al directorio de trabajo de Colab. Si faltan bibliotecas, instala `numpy`, `pandas`, `matplotlib`, `seaborn` y `scikit-learn` con `%pip install numpy pandas matplotlib seaborn scikit-learn` y reinicia el entorno antes de ejecutar desde el principio.

Para usarlo localmente, se recomienda Python 3.12. Desde la raíz del repositorio:

```bash
python -m pip install -r Practicas/requirements_practica_02.txt jupyterlab
jupyter lab Practicas/practica_02_preprocesamiento_riesgo_crediticio.ipynb
```

La carga local encuentra el CSV desde la raíz del repositorio o desde `Practicas/` y no necesita conexión a internet. El notebook incluye las salidas y las gráficas guardadas, por lo que también puede revisarse sin ejecutarlo.

Para la entrega, comparte el enlace del repositorio y sube `Practicas/practica_02_preprocesamiento_riesgo_crediticio.ipynb` a la plataforma del curso. Si entregas también el CSV, conserva la atribución de su fuente.
