# Diccionario de variables

El archivo `heart.csv` contiene 13 predictores y una variable objetivo binaria. Las columnas están almacenadas numéricamente, pero algunas representan categorías codificadas.

| Variable | Rol / tipo analítico | Interpretación |
|---|---|---|
| `age` | Cuantitativa | Edad en años. |
| `sex` | Binaria | Sexo codificado en el dataset. |
| `cp` | Categórica codificada | Tipo de dolor torácico. |
| `trestbps` | Cuantitativa | Presión arterial en reposo (mm Hg). |
| `chol` | Cuantitativa | Colesterol sérico (mg/dl). |
| `fbs` | Binaria | Indicador de glucosa en ayunas elevada. |
| `restecg` | Categórica codificada | Resultado del electrocardiograma en reposo. |
| `thalach` | Cuantitativa | Frecuencia cardíaca máxima alcanzada. |
| `exang` | Binaria | Angina inducida por ejercicio. |
| `oldpeak` | Cuantitativa | Depresión del segmento ST inducida por ejercicio respecto al reposo. |
| `slope` | Categórica codificada | Pendiente del segmento ST en máximo esfuerzo. |
| `ca` | Conteo discreto | Número de vasos principales observados/coloreados. |
| `thal` | Categórica codificada | Resultado codificado de la prueba `thal`. |
| `target` | Dependiente binaria | Clase objetivo: 0/1 según la codificación del dataset. |

## Decisión de modelado

`cp`, `restecg`, `slope` y `thal` se tratan como categorías mediante variables indicadoras. `sex`, `fbs` y `exang` se mantienen como indicadores binarios. `ca` se conserva como conteo discreto. Esta decisión evita imponer artificialmente una distancia lineal entre códigos categóricos.

## Fuente

Dataset de Kaggle conocido como **Heart Disease Dataset** (`johnsmith88/heart-disease-dataset`), basado en los atributos del conjunto Heart Disease de UCI. El archivo usado en este proyecto fue proporcionado directamente como `heart.csv`.
