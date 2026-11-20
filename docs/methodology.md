# Notas metodológicas

## Separar inferencia de predicción

Este proyecto usa dos implementaciones complementarias de regresión logística porque responden a preguntas distintas:

- **statsmodels**: inferencia, coeficientes, errores estándar, `z`, p-values, intervalos de confianza, likelihood y pseudo-R²;
- **scikit-learn**: evaluación predictiva reproducible sobre un conjunto de prueba separado.

No se reporta el accuracy sobre las mismas filas utilizadas para estimar el modelo inferencial como si fuera desempeño futuro.

## Duplicados

El CSV tiene 1,025 filas y 723 son duplicados exactos. Después de `drop_duplicates()` quedan 302 observaciones. La decisión de modelar las filas únicas reduce el riesgo de pseudorreplicación y de inferencia excesivamente optimista.

## Variables categóricas codificadas

`cp`, `restecg`, `slope` y `thal` son códigos categóricos. En el modelo inferencial se especifican con `C(variable)` para que statsmodels genere indicadores. De esta forma, cada nivel se compara contra una categoría de referencia y no se impone una pendiente lineal arbitraria sobre el código.

## Variables raras

Algunos códigos tienen muy pocas observaciones (`restecg=2`, `thal=0`, `ca=4`). Se conservan para respetar los datos recibidos, pero los intervalos de confianza asociados a niveles raros pueden ser amplios. Por eso la interpretación no se limita a p-values individuales: también se calculan pruebas de razón de verosimilitudes para cada variable fuente completa.

## Pseudo-R²

Se usa el pseudo-R² de McFadden:

\[
R^2_{McFadden} = 1 - \frac{\log L_{modelo}}{\log L_{nulo}}
\]

No debe interpretarse como el porcentaje de varianza explicada de una regresión lineal. Es una medida relativa de mejora en log-verosimilitud respecto al modelo con solo intercepto.

## LLR p-value

La prueba de razón de verosimilitudes compara el modelo con predictores frente al modelo nulo. Un p-value pequeño indica que, como conjunto, los términos incluidos mejoran el ajuste de manera estadísticamente detectable.

Para la búsqueda de modelos reducidos, si dos candidatos tuvieran p-values prácticamente indistinguibles, se usa el estadístico LLR y después AIC como desempate determinista.

## Curva logística en dos dimensiones

Una regresión logística multivariable no puede representarse completamente con una sola curva sigmoide 2D. Por eso `logistic_curve_oldpeak.png` es una visualización **univariada complementaria** de `oldpeak` contra `target`. El modelo multivariable se evalúa con sus métricas y tablas, no mediante esa curva simplificada.
