import streamlit as tf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Configuración de la página del Dashboard
st.set_page_config(
    page_title="Plataforma de Analítica - Detección de Fraude Académico",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Sistema Integral de Analítica y Detección de Fraude Académico")
st.markdown("### Monitoreo en Tiempo Real y Modelado Predictivo (PMV3)")
st.write("---")

# 1. Carga de Datos
@st.cache_data
def load_data():
    # Intenta leer el archivo localmente (asegúrate de subir el CSV con este nombre a tu GitHub)
    try:
        df = pd.read_csv("part-00000-masivo (2).csv")
        return df
    except FileNotFoundError:
        st.error("No se encontró el archivo 'part-00000-masivo (2).csv'. Por favor, súbelo a tu repositorio de GitHub.")
        return None

df = load_data()

if df is not None:
    # --- SECCIÓN DE ENTRENAMIENTO DEL MODELO PREDICTIVO ---
    # Separamos variables para entrenar un modelo rápido y mostrar métricas predictivas reales
    X = df[['cambios_pestana', 'tiempo_examen_min', 'score_biometria', 'similitud_texto_porcentaje', 'intensidad_cambios']]
    y = df['alerta_fraude']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # --- SECCIÓN 1: 4 CARDS DE KPIs ---
    st.subheader("📊 Indicadores Clave de Rendimiento (KPIs)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_estudiantes = len(df)
    alertas_totales = int(df['alerta_fraude'].sum())
    porcentaje_fraude = (alertas_totales / total_estudiantes) * 100
    promedio_cambios = df['cambios_pestana'].mean()
    
    with kpi1:
        st.metric(label="Total Estudiantes Evaluados", value=f"{total_estudiantes:,}")
    with kpi2:
        st.metric(label="Alertas de Fraude Detectadas", value=f"{alertas_totales:,}", delta=f"{porcentaje_fraude:.1f}% del total", delta_color="inverse")
    with kpi3:
        st.metric(label="Promedio Cambios de Pestaña", value=f"{promedio_cambios:.2f}")
    with kpi4:
        st.metric(label="Similitud de Texto Promedio", value=f"{df['similitud_texto_porcentaje'].mean():.1f}%")
        
    st.write("---")
    
    # --- SECCIÓN 2: 3 GRÁFICOS DESCRIPTIVOS (HISTÓRICOS) ---
    st.subheader("📈 Gráficos de Comportamiento Histórico")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Distribución de Alertas de Fraude**")
        fig_pie = px.pie(df, names='alerta_fraude', title="Proporción de Fraude vs Normal", 
                         labels={'alerta_fraude':'Alerta'}, color='alerta_fraude',
                         color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.markdown("**Intensidad de Cambios vs Tiempo de Examen**")
        fig_scatter = px.scatter(df, x='tiempo_examen_min', y='intensidad_cambios', 
                                 color='alerta_fraude', title="Dispersión por Intensidad de Alertas",
                                 color_continuous_scale=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col3:
        st.markdown("**Distribución del Score de Biometría**")
        fig_hist = px.histogram(df, x='score_biometria', color='alerta_fraude', 
                                barmode='overlay', title="Score Biométrico según Estado",
                                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        st.plotly_chart(fig_hist, use_container_width=True)
        
    st.write("---")
    
    # --- SECCIÓN 3: 3 GRÁFICOS PREDICTIVOS / AVANZADOS ---
    st.subheader("🤖 Análisis Predictivo e Inteligencia Artificial")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("**Importancia de Características (Random Forest)**")
        importances = model.feature_importances_
        features = X.columns
        df_imp = pd.DataFrame({'Variable': features, 'Importancia': importances}).sort_values(by='Importancia', ascending=True)
        fig_imp = px.bar(df_imp, x='Importancia', y='Variable', orientation='h',
                         title="¿Qué variables influyen más en la predicción?", color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col5:
        st.markdown("**Matriz de Riesgo Predictivo**")
        # Creamos una simulación de rangos de probabilidad
        df['Probabilidad_Fraude'] = model.predict_proba(X)[:, 1]
        fig_box = px.box(df, x='alerta_fraude', y='Probabilidad_Fraude', 
                         title="Nivel de Certeza del Modelo IA", color='alerta_fraude',
                         color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col6:
        st.markdown("**Simulador de Alerta en Tiempo Real (Predicción Activa)**")
        # Formulario interactivo para que el usuario ingrese datos y el modelo prediga
        with st.form("predict_form"):
            val_cambios = st.number_input("Cambios de Pestaña", min_value=0, max_value=100, value=5)
            val_tiempo = st.number_input("Tiempo Examen (Min)", min_value=1, max_value=180, value=60)
            val_bio = st.slider("Score Biometría", min_value=0.0, max_value=1.0, value=0.8)
            val_sim = st.slider("% Similitud Texto", min_value=0.0, max_value=100.0, value=20.0)
            val_int = st.number_input("Intensidad de Cambios", min_value=0.0, max_value=20.0, value=1.5)
            
            submit = st.form_submit_values = st.form_submit_button("Analizar Comportamiento")
            
            if submit:
                input_data = [[val_cambios, val_tiempo, val_bio, val_sim, val_int]]
                prediccion = model.predict(input_data)[0]
                probabilidad = model.predict_proba(input_data)[0][1]
                
                if prediccion == 1:
                    st.error(f"🚨 ALERTA: Alta Probabilidad de Fraude ({probabilidad*100:.1f}%)")
                else:
                    st.success(f"✅ COMPORTAMIENTO NORMAL: Probabilidad de Fraude ({probabilidad*100:.1f}%)")