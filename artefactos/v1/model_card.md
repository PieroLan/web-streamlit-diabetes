# Model Card — LRN
**Versión:** v1  
**Sistema:** Python 3.12.7, scikit-learn 1.5.1

## Datos
Archivo: `df_sin_outliers_categorizado.csv` | Shape: (2379, 11) | Objetivo: `QS109` (No-tiene=0, Tiene=1) | Prevalencia: 0.454

## Entrenamiento
Split 80/20 estratificado (random_state=42). Preprocesamiento: StandardScaler (num) + OneHotEncoder(ignore) (cat) + SMOTE(k=3).

## Modelo
Seleccionado para TEST: **LRN**.  
Umbral de decisión: **0.50** (provisional).

## Métricas en TEST
F1=0.706, P=0.718, R=0.694,
ROC-AUC=0.823, PR-AUC=0.789.
