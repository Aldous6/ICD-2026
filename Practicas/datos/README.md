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
