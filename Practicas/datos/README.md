# Datos de la práctica 02

`default_credit_card_clients.csv` es una copia sin modificaciones del [CSV publicado por UCI](https://archive.ics.uci.edu/static/public/350/data.csv) para [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). Se descargó el 4 de octubre de 2026. Tiene 30 000 registros y 25 columnas: un identificador, 23 atributos explicativos y la variable objetivo.

**Referencia:** Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C55S3H](https://doi.org/10.24432/C55S3H).

**Licencia:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), indicada en la ficha de UCI.

**SHA-256 del archivo original:**

```text
45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0
```

Se mantienen los encabezados originales `ID`, `X1` a `X23` y `Y`. El notebook los cambia por nombres descriptivos en memoria y conserva una copia de los valores originales. Las etiquetas de categorías y las características extraídas no se escriben sobre este archivo.

Los datos corresponden a clientes de tarjetas de crédito de Taiwán, con historial de abril a septiembre de 2005. Los importes están en dólares de Taiwán (NT$). La descripción de las variables se consulta en la ficha de UCI; los códigos que esa ficha no explica se señalan como no documentados en el notebook.

# Datos de la práctica 03

La práctica utiliza la división proporcionada por el curso de [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697), de UCI. El conjunto público contiene 4424 estudiantes y 36 atributos; su objetivo describe la situación del estudiante al final de la duración normal de su carrera. Incluye información de inscripción y desempeño académico de los dos primeros semestres.

**Referencia:** Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5MC89](https://doi.org/10.24432/C5MC89).

**Licencia del conjunto de UCI:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). La división se tomó del [repositorio del curso](https://github.com/husseinlopez/icd2026/tree/main/datos), sin reconstruirla a partir de UCI.

`desercion-entrenamiento.csv` contiene 3539 estudiantes y 38 columnas: `id`, los 36 atributos y `Target`. Las clases son `Graduate` (1767), `Dropout` (1137) y `Enrolled` (635). `desercion-prueba.csv` contiene 885 estudiantes y 31 columnas: `id` y los 30 atributos conocidos al cierre del primer semestre, sin `Target` ni los seis atributos del segundo semestre. Ambos archivos tienen identificadores únicos y carecen de valores faltantes. No hay identificadores compartidos entre ellos; en conjunto abarcan 4424 estudiantes.

Los archivos se descargaron el **4 de noviembre de 2026** y se conservaron sin modificar sus bytes. Sus fuentes directas y huellas SHA-256 son:

| Archivo | Fuente | SHA-256 |
| --- | --- | --- |
| `desercion-entrenamiento.csv` | [CSV del curso](https://raw.githubusercontent.com/husseinlopez/icd2026/main/datos/desercion-entrenamiento.csv) | `7c477f6b0514318efcc793e69418b7d3e0db5e660eeb5a7d0370f3dfd6e941f8` |
| `desercion-prueba.csv` | [CSV del curso](https://raw.githubusercontent.com/husseinlopez/icd2026/main/datos/desercion-prueba.csv) | `c7ba06a34d00a7c8b19a00908760c75914344d18718b25ecb5db78d6904c48b1` |

El notebook registra las huellas durante la ejecución para identificar los datos utilizados. Las estadísticas de esta sección se comprobaron directamente sobre estas copias.

Se deben conservar los archivos originales y sus identificadores. El notebook obtiene la distribución de clases del entrenamiento, interpreta los códigos nominales conforme a la ficha de UCI y usa exclusivamente los atributos disponibles en prueba para el modelo del primer semestre. Los atributos del segundo semestre se reservan para la comparación adicional dentro del entrenamiento. El CSV generado de predicciones se guarda fuera de esta carpeta y no modifica los datos originales.

La versión refinada del [notebook](../practica_03_clasificacion_desercion.ipynb) se ejecutó completa el 4 de noviembre de 2026 y conserva sus salidas. Incluye la comparación actualizada con el segundo semestre y generó [predicciones_desercion.csv](../predicciones_desercion.csv), con 885 filas y las columnas `id` y `Target`, cuyas etiquetas son `dropout`, `enrolled` o `graduate`. El SHA-256 del CSV vigente es `7342df54d15f7f9c7184e76b2428c50fe283b02a6ae8260ed928298acbcebc40`, registrado también en [metadatos.json](../resultados_practica_03/metadatos.json).

La [referencia inicial](../resultados_practica_03/referencia_inicial/) conserva las predicciones y los resultados anteriores al refinamiento. El CSV archivado `predicciones_iniciales.csv` tiene SHA-256 `22487ffd177e0a3b8f57b7eb1b4f2e9b60e49903de4dab631c14b31f0d0bdd3e`; [huellas.json](../resultados_practica_03/referencia_inicial/huellas.json) registra también las demás copias. Las predicciones no son las etiquetas verdaderas de prueba; su F1 se desconoce.

# Datos de la práctica 04: Bank Marketing

`bank-additional-full.csv` y `bank-additional-names.txt` son los miembros originales de `bank-additional.zip`, incluido en el [paquete oficial de UCI](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip). Se descargaron el 7 de noviembre de 2026 sin modificar sus bytes. La variante completa contiene 41,188 filas, 20 atributos y el objetivo `y`. No debe confundirse con `bank-full.csv`, de otra versión. La práctica conserva el archivo y elimina `duration` únicamente en memoria.

Referencia: Moro, S., Rita, P. y Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. [DOI: 10.24432/C5K306](https://doi.org/10.24432/C5K306). Licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), indicada en la [ficha de UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing). La descripción incluida documenta las variables y el orden temporal; se conserva `unknown` como categoría original.

SHA-256 de `bank-additional-full.csv`: `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`.
