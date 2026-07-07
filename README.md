# Proyecto 1: Análisis Exploratorio de Datos (EDA) - Titanic

## 📌 Objetivo del proyecto
Analizar el dataset histórico del Titanic para identificar los factores
que se relacionan con la supervivencia de los pasajeros, como primer paso
antes de construir un modelo predictivo (Semana 2 del curso).

## 🎯 Problema de negocio (simulado)
Aunque este es un dataset histórico, el ejercicio simula un caso real de
análisis de riesgo: identificar qué variables (sexo, clase social, edad,
tarifa pagada) se asocian con un resultado (supervivencia), tal como se
haría en un análisis de riesgo de crédito, abandono de clientes, o
diagnóstico médico.

## 📂 Estructura del proyecto
```
proyecto1_titanic/
├── data/
│   └── titanic.csv          # dataset original (891 pasajeros, 12 columnas)
├── graficas/
│   └── eda_titanic.png       # visualizaciones generadas
├── eda_titanic.py             # script principal del análisis
└── README.md                   # este archivo
```

## 🛠️ Herramientas utilizadas
- Python 3
- Pandas (carga, exploración, agregación de datos)
- NumPy (operaciones vectorizadas)
- Matplotlib (visualización)

## 🔍 Metodología
1. Carga de datos con `pd.read_csv()`
2. Exploración inicial: `.head()`, `.info()`, `.describe()`
3. Detección de valores nulos por columna
4. Análisis de la variable objetivo (`Survived`)
5. Análisis cruzado por sexo y clase social (`.groupby()`)
6. Análisis estadístico de edad y tarifa (media, mediana, desviación estándar)
7. Detección de outliers comparando media vs. mediana
8. Visualización de todos los hallazgos

## 📊 Hallazgos principales

| Hallazgo | Dato |
|---|---|
| Tasa general de supervivencia | 38.4% |
| Supervivencia mujeres | 74.2% |
| Supervivencia hombres | 18.9% |
| Supervivencia 1ra clase | 63.0% |
| Supervivencia 3ra clase | 24.2% |
| Valores nulos en Edad | 19.9% del dataset |
| Valores nulos en Cabina | 77.1% del dataset |
| Outliers detectados en Fare | Media ($32.2) muy superior a mediana ($14.45) |

## 💡 Conclusiones
- **El sexo es el predictor más fuerte** de supervivencia en este dataset,
  reflejando la política histórica de evacuación "mujeres y niños primero".
- **La clase social importa**: pasajeros de 1ra clase sobrevivieron a más
  del doble de tasa que los de 3ra clase, probablemente por la ubicación
  de sus camarotes respecto a los botes salvavidas.
- **Hay datos faltantes que requieren tratamiento**: la columna `Age` tiene
  19.9% de nulos (manejable con imputación); `Cabin` tiene 77.1% de nulos
  (probablemente se descartará o se usará solo su presencia/ausencia).
- **La tarifa (`Fare`) tiene outliers fuertes**, visibles en la gran
  diferencia entre su media y su mediana - candidata a transformación
  logarítmica antes de usarla en un modelo (técnica vista en Día 1: los
  logaritmos "comprimen" diferencias extremas).

## 🔜 Próximos pasos (Semana 2)
Este EDA sienta las bases para entrenar un modelo de clasificación
(Regresión Logística, Árboles de Decisión) que prediga la supervivencia
de un pasajero según estas variables.

## 👤 Autor
Johnnoel - Proyecto 1 del curso intensivo de Machine Learning (Día 7)
