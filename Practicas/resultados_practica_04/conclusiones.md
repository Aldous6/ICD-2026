En deserción ganó **LightGBM**, con F1 macro 0.6968; la mejor base fija fue SVM (0.6918). La diferencia fue +0.49 puntos porcentuales. Bagging 400 cambió el F1 del árbol sin límite en +6.25 puntos; AdaBoost cambió el del tocón en +14.68 puntos.

Las correlaciones de error entre las cinco bases fueron de 0.452 a 0.863, con media 0.559. Son errores OOF de estas muestras; la complementariedad observada debe interpretarse junto a la calidad de cada base.

En Bank ganó **Umbral interno: Transferido: LightGBM**, con F1 macro 0.7180 y sensibilidad yes 0.5532. Transferir la configuración de LightGBM produjo 0.6959: la diferencia del mejor procedimiento Bank fue +2.21 puntos. La tabla permite distinguir transferir la configuración, volver a optimizarla y cambiar de familia; no se afirma que una familia sea universalmente superior.

La familia ganadora coincidió en ambos conjuntos. La configuración transferida se reajustó con datos bancarios; una coincidencia de familia no implica que sirvan las mismas variables, pesos o umbral.

Random Forest: en deserción obtuvo 0.6960 (+0.42 puntos frente a SVM); en Bank 0.7153 (+4.13 puntos frente a Regresión logística balanceada). Extra-Trees: en deserción obtuvo 0.6907 (-0.12 puntos frente a SVM); en Bank 0.7068 (+3.29 puntos frente a Regresión logística balanceada). LightGBM: en deserción obtuvo 0.6968 (+0.49 puntos frente a SVM); en Bank 0.6935 (+1.95 puntos frente a Regresión logística balanceada). CatBoost: en deserción obtuvo 0.6916 (-0.02 puntos frente a SVM); en Bank 0.7173 (+4.33 puntos frente a Regresión logística balanceada). Votación suave: en deserción obtuvo 0.6690 (-2.28 puntos frente a SVM); en Bank 0.7177 (+4.37 puntos frente a Regresión logística balanceada). Stacking logístico: en deserción obtuvo 0.6717 (-2.01 puntos frente a SVM); en Bank 0.6751 (+0.12 puntos frente a Regresión logística balanceada). En deserción las combinaciones usan los cinco base; en Bank usan las tres familias optimizadas con mayor F1. Esta diferencia debe considerarse al compararlas.

En Bank, CatBoost con categorías nativas obtuvo 0.7163, una diferencia de -0.09 puntos frente a su representación codificada, manteniendo los hiperparámetros del clasificador.

No se alcanzó el criterio de meseta local en deserción/catboost, deserción/lightgbm, Bank/extra_trees, Bank/lightgbm; 50 configuraciones no bastan para afirmar estabilización en esas búsquedas. La meseta es local al muestreo y no demuestra optimalidad global.

En deserción, enrolled representa 17.94% de las filas; en Bank, yes representa 11.27%. El primero distingue tres situaciones académicas y el segundo una respuesta binaria menos frecuente. Son diferencias compatibles con cambios en pesos, sensibilidad y umbral; las tablas no demuestran causalmente por qué ganó un modelo. Bank también incluye indicadores económicos y contactos previos, mientras deserción contiene rendimiento académico.

Las búsquedas comparten los pliegues de la tabla: elegir entre configuraciones y familias introduce optimismo. Las desviaciones no prueban significación. Bank viene ordenado por fecha, pero la consigna exige particiones aleatorias estratificadas; este resultado no demuestra rendimiento en campañas futuras. No se conocen las etiquetas de prueba de deserción ni se garantiza un lugar en la clasificación del curso.
