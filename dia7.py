"""
PROYECTO 1 - Semana 1: EDA (Exploratory Data Analysis) del dataset Titanic
Objetivo: aplicar TODO lo aprendido en la Semana 1 (Días 1-7) a un dataset real.

Estructura profesional de proyecto:
proyecto1_titanic/
├── data/
│   └── titanic.csv
├── eda_titanic.py       <- este archivo
├── graficas/            <- se genera automáticamente
└── README.md
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 0. Configuración
# ============================================================
os.makedirs("graficas", exist_ok=True)
pd.set_option("display.width", 120)

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
print("="*60)
print("1. CARGA DE DATOS")
print("="*60)

df = pd.read_csv("data/titanic.csv")
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas\n")
print("Primeras filas:")
print(df.head())

# ============================================================
# 2. EXPLORACIÓN INICIAL
# ============================================================
print("\n" + "="*60)
print("2. EXPLORACIÓN INICIAL (df.info, df.describe)")
print("="*60)

print("\nTipos de datos y valores no nulos:")
print(df.info())

print("\nEstadísticas descriptivas (columnas numéricas):")
print(df.describe())

# ============================================================
# 3. VALORES NULOS (primer contacto - lo profundizamos Semana 2)
# ============================================================
print("\n" + "="*60)
print("3. VALORES NULOS POR COLUMNA")
print("="*60)

nulos = df.isnull().sum()
porcentaje_nulos = (nulos / len(df) * 100).round(2)
tabla_nulos = pd.DataFrame({"nulos": nulos, "porcentaje": porcentaje_nulos})
print(tabla_nulos[tabla_nulos["nulos"] > 0].sort_values("nulos", ascending=False))

# ============================================================
# 4. ANÁLISIS DE LA VARIABLE OBJETIVO: Survived
# ============================================================
print("\n" + "="*60)
print("4. ANÁLISIS DE SUPERVIVENCIA (variable objetivo)")
print("="*60)

tasa_supervivencia = df["Survived"].mean()
print(f"Tasa general de supervivencia: {tasa_supervivencia:.2%}")
print(f"Sobrevivieron: {df['Survived'].sum()} de {len(df)} pasajeros")

# ============================================================
# 5. SUPERVIVENCIA POR SEXO (groupby - reemplaza el código del Día 5)
# ============================================================
print("\n" + "="*60)
print("5. SUPERVIVENCIA POR SEXO")
print("="*60)

supervivencia_sexo = df.groupby("Sex")["Survived"].mean()
print(supervivencia_sexo)
print("\n-> Nota la diferencia enorme: esto ya es una pista fuerte de que 'Sex'")
print("   será una variable MUY importante cuando entrenemos un modelo (Semana 2).")

# ============================================================
# 6. SUPERVIVENCIA POR CLASE (Pclass: 1ra, 2da, 3ra clase)
# ============================================================
print("\n" + "="*60)
print("6. SUPERVIVENCIA POR CLASE SOCIAL")
print("="*60)

supervivencia_clase = df.groupby("Pclass")["Survived"].mean()
print(supervivencia_clase)

# ============================================================
# 7. ANÁLISIS DE EDAD (usando estadísticas del Día 2)
# ============================================================
print("\n" + "="*60)
print("7. ANÁLISIS DE EDAD")
print("="*60)

print(f"Edad media: {df['Age'].mean():.1f} años")
print(f"Edad mediana: {df['Age'].median():.1f} años")
print(f"Desviación estándar: {df['Age'].std():.1f} años")
print(f"Edad mínima: {df['Age'].min()}, máxima: {df['Age'].max()}")

# ¿Por qué usar mediana también? (recuerda Día 2: outliers)
diferencia = abs(df['Age'].mean() - df['Age'].median())
print(f"\nDiferencia media-mediana: {diferencia:.2f}")
print("-> Si esta diferencia fuera grande, sospecharíamos de outliers en la edad.")

# ============================================================
# 8. DETECCIÓN BÁSICA DE OUTLIERS EN 'Fare' (precio del ticket)
# ============================================================
print("\n" + "="*60)
print("8. OUTLIERS EN EL PRECIO DEL TICKET (Fare)")
print("="*60)

media_fare = df["Fare"].mean()
mediana_fare = df["Fare"].median()
print(f"Media de Fare: ${media_fare:.2f}")
print(f"Mediana de Fare: ${mediana_fare:.2f}")
print("-> Nota la GRAN diferencia entre media y mediana: señal clara de outliers")
print("   (algunos pasajeros pagaron tarifas muchísimo más altas que el resto).")

tickets_caros = df[df["Fare"] > 200]
print(f"\nPasajeros que pagaron más de $200: {len(tickets_caros)}")

# ============================================================
# 9. VISUALIZACIONES
# ============================================================
print("\n" + "="*60)
print("9. GENERANDO VISUALIZACIONES")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Gráfica 1: Supervivencia por sexo
supervivencia_sexo.plot(kind="bar", ax=axes[0,0], color=["salmon", "steelblue"])
axes[0,0].set_title("Tasa de supervivencia por sexo")
axes[0,0].set_ylabel("Tasa de supervivencia")
axes[0,0].set_xticklabels(supervivencia_sexo.index, rotation=0)

# Gráfica 2: Supervivencia por clase
supervivencia_clase.plot(kind="bar", ax=axes[0,1], color="seagreen")
axes[0,1].set_title("Tasa de supervivencia por clase social")
axes[0,1].set_ylabel("Tasa de supervivencia")
axes[0,1].set_xticklabels(["1ra", "2da", "3ra"], rotation=0)

# Gráfica 3: Distribución de edades
axes[1,0].hist(df["Age"].dropna(), bins=30, color="orchid", edgecolor="black")
axes[1,0].axvline(df["Age"].mean(), color="red", linestyle="--", label=f"Media: {df['Age'].mean():.1f}")
axes[1,0].axvline(df["Age"].median(), color="blue", linestyle="--", label=f"Mediana: {df['Age'].median():.1f}")
axes[1,0].set_title("Distribución de edades")
axes[1,0].set_xlabel("Edad")
axes[1,0].legend()

# Gráfica 4: Distribución de tarifas (para ver los outliers)
axes[1,1].hist(df["Fare"], bins=40, color="goldenrod", edgecolor="black")
axes[1,1].axvline(df["Fare"].mean(), color="red", linestyle="--", label=f"Media: ${df['Fare'].mean():.1f}")
axes[1,1].axvline(df["Fare"].median(), color="blue", linestyle="--", label=f"Mediana: ${df['Fare'].median():.1f}")
axes[1,1].set_title("Distribución de tarifas (Fare) - nota los outliers")
axes[1,1].set_xlabel("Tarifa ($)")
axes[1,1].legend()

plt.tight_layout()
plt.savefig("graficas/eda_titanic.png", dpi=120)
print("Gráficas guardadas en graficas/eda_titanic.png")

# ============================================================
# 10. CONCLUSIONES DEL ANÁLISIS
# ============================================================
print("\n" + "="*60)
print("10. CONCLUSIONES CLAVE DEL EDA")
print("="*60)
print("""
1. El sexo es la variable más determinante de supervivencia observada.
2. La clase social (Pclass) tiene una relación clara con la supervivencia:
   a mejor clase, mayor tasa de supervivencia.
3. La columna 'Age' tiene valores nulos que habrá que tratar antes de
   entrenar cualquier modelo (Semana 2: imputación de valores nulos).
4. La columna 'Fare' tiene outliers fuertes (tickets muy caros que
   distorsionan la media) - candidata a transformación logarítmica
   o normalización antes de modelar (Semana 2).
5. Este EDA es la base para el proyecto de la Semana 2, donde
   entrenaremos nuestro primer modelo de clasificación real
   para predecir supervivencia.
""")