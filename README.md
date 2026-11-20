# Heart Disease · Logistic Regression Analysis

Análisis estadístico y predictivo de un dataset de enfermedad cardíaca mediante **regresión logística**. El proyecto está planteado como una práctica de portafolio: prioriza calidad de datos, interpretación inferencial, selección de modelos y evaluación fuera de muestra, en lugar de limitarse a entrenar un clasificador y reportar una métrica.

## Resultado principal

El archivo original contiene **1,025 registros**, pero **723 son duplicados exactos**. Para no inflar artificialmente el tamaño muestral, el análisis trabaja con **302 observaciones únicas**. Sobre esa muestra, el modelo logístico completo alcanza un **pseudo-R² de McFadden de 0.518** y un LLR p-value de aproximadamente **2.55e-35**.

En evaluación predictiva con un holdout estratificado 80/20 y semilla reproducible:

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Completo | 0.885 | 0.861 | 0.939 | 0.899 | 0.938 |
| Reducido | 0.836 | 0.811 | 0.909 | 0.857 | 0.910 |

La búsqueda exhaustiva de **1,092 modelos candidatos** con máximo cuatro variables selecciona el mismo conjunto por dos criterios distintos:

**`cp + oldpeak + ca + thal`**

- Menor LLR p-value: `4.49e-36`.
- Mayor pseudo-R² de McFadden entre los candidatos: `0.448`.
- El modelo reducido conserva buena capacidad discriminativa con menos variables.

> Este proyecto es un ejercicio analítico y educativo. No debe interpretarse como una herramienta clínica ni usarse para tomar decisiones médicas.

## Preguntas que aborda

El proyecto busca responder, entre otras, estas preguntas:

- ¿Qué problemas de calidad tiene el dataset antes de modelar?
- ¿Qué variables muestran evidencia estadística después de controlar por las demás?
- ¿Cómo se interpretan los coeficientes logísticos y los *odds ratios*?
- ¿Qué tan superior es el modelo completo frente a un modelo nulo?
- ¿Qué combinación de hasta cuatro variables produce el mejor modelo reducido?
- ¿Cuánto desempeño se pierde al simplificar el modelo?
- ¿Cómo cambia la probabilidad estimada de `target=1` con una variable continua como `oldpeak`?

## Estructura del repositorio

```text
heart-disease-logistic-analysis/
├── data/
│   ├── raw/
│   │   └── heart.csv
│   ├── processed/
│   │   └── heart_clean.csv
│   └── README.md
├── docs/
│   ├── methodology.md
│   └── variable_dictionary.md
├── notebooks/
│   ├── 01_data_audit_and_eda.ipynb
│   ├── 02_full_logistic_model.ipynb
│   └── 03_model_selection_and_evaluation.ipynb
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   ├── config.py
│   ├── data.py
│   ├── modeling.py
│   ├── selection.py
│   ├── visualization.py
│   └── run_analysis.py
├── tests/
│   └── test_pipeline.py
├── run_analysis.py
├── requirements.txt
└── pyproject.toml
```

## Metodología

### 1. Auditoría y limpieza

Se valida el esquema, se cuantifican faltantes y duplicados y se elimina únicamente la duplicación exacta. No hay celdas faltantes en el archivo recibido.

La eliminación de duplicados es importante aquí porque repetir exactamente una observación no aporta nueva evidencia independiente y puede hacer que los errores estándar parezcan artificialmente pequeños.

### 2. Tratamiento de variables

Todas las columnas vienen almacenadas como números, pero no todas representan magnitudes continuas. El modelo inferencial trata como categorías:

- `cp`
- `restecg`
- `slope`
- `thal`

Los indicadores binarios `sex`, `fbs` y `exang` se conservan como 0/1; `ca` se interpreta como conteo discreto. Esta especificación evita asumir, por ejemplo, que pasar de una categoría codificada 0 a 1 tiene exactamente el mismo efecto que pasar de 1 a 2.

### 3. Regresión logística inferencial

El modelo se estima con `statsmodels` por máxima verosimilitud. Se reportan:

- coeficientes;
- errores estándar;
- estadísticos `z`;
- p-values;
- intervalos de confianza del 95%;
- odds ratios;
- pseudo-R² de McFadden;
- Log-Likelihood;
- LL-Null;
- LLR y LLR p-value.

Además, las variables categóricas se evalúan como bloques mediante pruebas de razón de verosimilitudes entre modelos anidados.

### 4. Evaluación predictiva

Para no confundir ajuste dentro de muestra con capacidad predictiva, la matriz de confusión y las métricas se calculan en un **holdout estratificado del 20%**. El pipeline predictivo usa One-Hot Encoding para variables categóricas y estandarización de las variables cuantitativas.

### 5. Selección exhaustiva de modelos reducidos

Se prueban todas las combinaciones de una a cuatro variables fuente de los 13 predictores originales. En total se ajustan **1,092 modelos**. Se construyen dos rankings:

1. menor `LLR p-value`;
2. mayor pseudo-R² de McFadden.

Los dos criterios seleccionan `cp + oldpeak + ca + thal` en esta muestra.

## Variables con mayor evidencia en el modelo completo

Las pruebas de razón de verosimilitudes por variable fuente identifican como significativas al 5%:

| Variable | LR statistic | p-value |
|---|---:|---:|
| `ca` | 18.864 | 0.000014 |
| `cp` | 21.209 | 0.000095 |
| `sex` | 9.226 | 0.002387 |
| `thal` | 12.741 | 0.005232 |
| `oldpeak` | 4.673 | 0.030632 |

En la tabla de coeficientes, `ca` presenta el mayor valor absoluto de `z` entre los términos individuales del modelo completo (`z ≈ -4.118`).

## Cómo ejecutar el proyecto

### Windows / PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_analysis.py
```

Después, para abrir los notebooks:

```powershell
jupyter lab
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
jupyter lab
```

Los resultados se regeneran en `outputs/tables/` y `outputs/figures/`.

## Pruebas

```bash
pytest -q
```

Las pruebas verifican, entre otras cosas, que la limpieza produzca 302 observaciones únicas y que el modelo logístico completo converja.

## Notebooks

**01 · Data audit and EDA**  
Revisa estructura, duplicados, balance de la variable objetivo y estadística descriptiva.

**02 · Full logistic model**  
Desarrolla el modelo completo, interpretación de coeficientes, incertidumbre, odds ratios, pseudo-R², likelihood y significancia.

**03 · Model selection and evaluation**  
Realiza la búsqueda exhaustiva de modelos de hasta cuatro predictores, compara criterios, evalúa el holdout y presenta matriz de confusión, ROC y curva logística.

## Fuente de datos

El archivo corresponde al dataset de Kaggle conocido como **Heart Disease Dataset** (`johnsmith88/heart-disease-dataset`), basado en atributos del conjunto Heart Disease de UCI. En este repositorio se conserva exactamente el CSV recibido como `data/raw/heart.csv` y la versión depurada se genera de forma reproducible.

## Stack

Python · pandas · NumPy · SciPy · statsmodels · scikit-learn · Matplotlib · Jupyter
