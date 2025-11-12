# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================================
# 1. Cargar modelo entrenado
# =====================================================
MODEL_PATH = Path("artefactos") / "v1/pipeline_LRN.joblib"
modelo = joblib.load(MODEL_PATH)

st.title("🤖 Predicción de Riesgo de Salud")
st.write("Basado en variables de hábitos y condiciones personales (Cuestionario QS).")

# Tabs
tab1, tab2, tab3 = st.tabs(["🧪 Predicción", "📊 Análisis del modelo", "📈 Gráficos interactivos"])

# =====================================================
# --- TAB 1: Predicción individual
# =====================================================
with tab1:
    st.subheader("Predicción personalizada")

    st.write("Ingrese los valores para realizar la predicción del modelo:")

    # Entrada de datos según tus variables
    col1, col2 = st.columns(2)
    with col1:
        QS23 = st.slider("Edad (años)", 10, 100, 30)
        QSSEXO = st.selectbox("Sexo", ["Masculino", "Femenino"])
        QS207C = st.slider("Edad en que tomó por primera vez bebidas alcohólicas", 5, 60, 18)
        QS208 = st.selectbox("¿Ha consumido alcohol en los últimos 12 meses?", ["Sí", "No"])
        QS213C = st.slider("Días a la semana que consume frutas", 0, 7, 3)
        QS219C = st.slider("Días a la semana que consume verduras", 0, 7, 3)

    with col2:
        QS900 = st.slider("Peso (kg)", 30.0, 150.0, 70.0)
        QS901 = st.slider("Talla (cm)", 120.0, 210.0, 170.0)
        QS907 = st.slider("Perímetro abdominal (cm)", 50.0, 150.0, 90.0)
        QS102 = st.selectbox("¿Tiene hipertensión?", ["Sí", "No"])
        QS500 = st.selectbox("¿Tiene tos con flema?", ["Sí", "No"])

    # Crear dataframe de entrada
    entrada = pd.DataFrame([{
        'QS23': QS23,
        'QSSEXO': QSSEXO,
        'QS207C': QS207C,
        'QS208': QS208,
        'QS213C': QS213C,
        'QS219C': QS219C,
        'QS900': QS900,
        'QS901': QS901,
        'QS907': QS907,
        'QS102': QS102,
        'QS500': QS500
    }])

    # Botón para predecir
    if st.button("Predecir"):
        try:
            pred = modelo.predict(entrada)[0]
            prob = modelo.predict_proba(entrada)[0][1]
            resultado = "🩺 Tiene diabetes" if pred == 1 else "✅ No tiene diabetes"

            st.success(f"**Resultado:** {resultado}")
            st.write(f"Probabilidad estimada: **{prob:.2f}**")
        except Exception as e:
            st.error(f"Error al predecir: {e}")

# =====================================================
# --- TAB 2: Análisis del modelo
# =====================================================
with tab2:
    st.subheader("Análisis del modelo")

    try:
        coef_df = pd.DataFrame({
            'Variable': modelo.feature_names_in_,
            'Peso': modelo.coef_[0]
        }).sort_values(by='Peso', key=abs, ascending=False)

        st.write("Importancia de cada variable en la predicción")
        st.bar_chart(coef_df.set_index("Variable"))
    except Exception:
        st.warning("No se pudieron obtener los coeficientes del modelo. "
                   "Esto puede ocurrir si es un modelo no lineal (p. ej., árbol o random forest).")

# =====================================================
# --- TAB 3: Gráficos interactivos
# =====================================================
with tab3:
    st.subheader("📈 Exploración visual de variables")

    # Simulación de dataset (si no tienes aún dataset real)
    st.info("Se muestra un dataset simulado para fines demostrativos.")
    import numpy as np
    np.random.seed(42)

    df_sim = pd.DataFrame({
        'QS23': np.random.randint(15, 80, 200),
        'QSSEXO': np.random.choice(['Masculino', 'Femenino'], 200),
        'QS207C': np.random.randint(10, 50, 200),
        'QS208': np.random.choice(['Sí', 'No'], 200),
        'QS213C': np.random.randint(0, 7, 200),
        'QS219C': np.random.randint(0, 7, 200),
        'QS900': np.random.normal(70, 15, 200),
        'QS901': np.random.normal(165, 10, 200),
        'QS907': np.random.normal(90, 15, 200),
        'QS102': np.random.choice(['Sí', 'No'], 200),
        'QS500': np.random.choice(['Sí', 'No'], 200),
        'Riesgo': np.random.choice(['Positivo', 'Negativo'], 200)
    })

    # Selector de variables
    variables_numericas = ['QS23', 'QS207C', 'QS213C', 'QS219C', 'QS900', 'QS901', 'QS907']
    x_var = st.selectbox("Eje X", variables_numericas, index=0)
    y_var = st.selectbox("Eje Y", variables_numericas, index=1)

    fig = px.scatter(
        df_sim,
        x=x_var,
        y=y_var,
        color='Riesgo',
        hover_data=['QSSEXO', 'QS102', 'QS208'],
        title=f"{x_var} vs {y_var} según Riesgo",
        width=900,
        height=550
    )
    st.plotly_chart(fig)

    st.subheader("Distribución de peso según sexo")
    fig2 = px.histogram(df_sim, x='QS900', color='QSSEXO', nbins=30, barmode='overlay')
    st.plotly_chart(fig2, use_container_width=True)
