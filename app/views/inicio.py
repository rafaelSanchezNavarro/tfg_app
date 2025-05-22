import pandas as pd
import streamlit as st
from app.logic.load_data import (
    load_X_tsne_con_test,
    load_csv,
    load_model,
    load_y_test,
    load_X_tsne,
    load_y_train_class2,
    default_supervised_model,
    default_isolation_model,
    default_dbsan_model,
)
from app.logic.producer import start_simulated_traffic
from app.views import show_architecture_2, show_dbscan, show_isolation

def show():
    # === Estilos personalizados ===
    st.markdown("""
<style>
/* Fondo general */
html, body, .main {
    background-color: #FFFFFF;
}

/* Títulos y textos */
h1, h2, h3, .stMarkdown {
    color: #333333;
}

/* Contenedor principal */
.block-container {
    padding: 2rem;
    background-color: #FFFFFF;
}

/* Botones */
.stButton>button {
    background-color: #D5E8D4;
    color: black;
    border-radius: 6px;
    padding: 0.4rem 1rem;
    border: 1px solid #999999;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #BBBBBB;
    color: black;
}

/* Expanders */
details > summary {
    background-color: #DAE8FC !important;
    color: #000 !important;
    padding: 10px;
    border-radius: 8px;
    font-weight: bold;
    border: 1px solid #D5E8D4;
}
details[open] {
    background-color: #DAE8FC !important;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #D5E8D4;
    margin-bottom: 10px;
}

/* Checkbox y file uploader */
.stCheckbox>label {
    color: #000000;
    background-color: transparent;
    font-weight: 500;
    margin-left: 6px;
}
.stFileUploader {
    background-color: #DAE8FC;
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

    # === PRESENTACIÓN ROJA ===
    st.markdown("""
<div style="background-color:#F8CECC; padding:20px; border-radius:10px;">
    <h1 style="color:#000000;">📊 Trabajo de Fin de Grado</h1>
    <h3 style="margin-top:0;">Modelos de Aprendizaje Automático para la Detección de Intrusos en Sistemas IIoT</h3>
    <p><strong>Autor:</strong> Rafael Sánchez Navarro</p>
    <p><strong>Grado:</strong> Ingeniería Informática</p>
    <p><strong>Universidad:</strong> Universidad de Castilla-La Mancha – Escuela Superior de Ingeniería Informática</p>
</div>
<br/>
""", unsafe_allow_html=True)

    # === DESCRIPCIÓN AZUL ===
    st.markdown("""
<div style="background-color:#DAE8FC; padding:15px; border-radius:10px;">
    <h3 style="margin-bottom:0;">🧭 Descripción de la Aplicación</h3>
    <p style="margin-top:5px;">
        Esta herramienta ha sido desarrollada como parte de un Trabajo de Fin de Grado con el objetivo de facilitar el análisis de tráfico en entornos IIoT mediante técnicas de aprendizaje automático.
        Proporciona una interfaz interactiva y accesible para evaluar distintos modelos, tanto supervisados como no supervisados, aplicados a flujos de datos simulados o cargados por el usuario.
    </p>
</div>
<br/>
""", unsafe_allow_html=True)

    # === FUNCIONALIDADES ===
    st.markdown("### 🎯 Funcionalidades disponibles:")

    with st.expander("🔍 Clasificación Supervisada"):
        st.markdown("""
        Este módulo aplica modelos de clasificación previamente entrenados para detectar comportamientos anómalos en tiempo real.  
        La interfaz muestra las predicciones del modelo para cada muestra, junto con explicaciones generadas mediante LIME, facilitando la interpretación de los resultados.  
    
        Para su funcionamiento, el usuario debe cargar dos archivos:  
        - Un conjunto de prueba con las características (`X_test`) en formato `.csv`  
        - Las etiquetas reales correspondientes (`y_test`) también en formato `.csv`  
    
        También puede utilizar un modelo predefinido o cargar uno propio en formato `.pkl`.
        """)

    with st.expander("🧠 Clustering con DBSCAN"):
        st.markdown("""
        Este módulo aplica el algoritmo DBSCAN para detectar agrupaciones naturales en los datos sin necesidad de etiquetas previas.  
        Esta técnica es especialmente útil para identificar patrones anómalos en contextos no supervisados, donde no se conoce previamente qué es normal o anómalo.  
    
        Al finalizar el análisis, cada grupo (cluster) es evaluado: si más del 50% de sus elementos corresponden a anomalías conocidas, **todo el grupo se considera anómalo**.  
        A partir de esta clasificación binaria, se generan métricas de evaluación que permiten medir el rendimiento del modelo, incluyendo:
    
        - Tasa de acierto  
        - Precisión  
        - Sensibilidad (Recall)  
        - F1-score  
        - Matriz de confusión  
    
        Puedes utilizar un modelo predefinido o cargar uno propio en formato `.pkl`, junto con un conjunto de prueba en formato `.csv`.
        """)
    
    with st.expander("🚨 Detección de Anomalías con Isolation Forest"):
        st.markdown("""
        Este módulo aplica el algoritmo Isolation Forest para detectar observaciones que se desvían significativamente del comportamiento general.  
        Esta técnica es especialmente útil para identificar patrones anómalos en contextos no supervisados, donde no se conoce previamente qué es normal o anómalo.  
    
        Al finalizar el análisis, se asigna una clasificación binaria a cada observación (normal o anómala), y se generan métricas de evaluación que permiten medir el rendimiento del modelo, incluyendo:
    
        - Tasa de acierto  
        - Precisión  
        - Sensibilidad (Recall)  
        - F1-score  
        - Matriz de confusión  
    
        Puedes utilizar un modelo predefinido o cargar uno propio en formato `.pkl`, junto con un conjunto de prueba en formato `.csv`.
        """)

        
    # === SELECCIÓN ===
    st.markdown("### 🧪 Selecciona una sección para comenzar:")

    if "seleccion" not in st.session_state:
        st.session_state.seleccion = None

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔎 Clasificación"):
            st.session_state.seleccion = "clasificacion"
    with col2:
        if st.button("🧠 DBSCAN"):
            st.session_state.seleccion = "dbscan"
    with col3:
        if st.button("🚨 Isolation Forest"):
            st.session_state.seleccion = "isolation"

    st.markdown("---")

    # === Clasificación Supervisada ===
    if st.session_state.seleccion == "clasificacion":
        st.subheader("🔎 Clasificación Supervisada")

        use_default_model = st.checkbox("Utilizar modelo predeterminado")
        use_default_test = st.checkbox("Utilizar conjunto de prueba predeterminado")

        modelo_clasificacion = (
            default_supervised_model() if use_default_model
            else load_model(label="Cargar modelo de clasificación (.pkl)")
        )

        if use_default_test:
            X_test = None
            y_test = load_y_test()
        else:
            st.markdown("#### Cargar conjunto de prueba manual")
            X_test = load_csv(label="Archivo de entrada `X_test` (.csv)")
            y_test = load_csv(label="Archivo de etiquetas `y_test` (.csv)")
            if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
                y_test = y_test.iloc[:, 0]

        if "producer_started" not in st.session_state:
            st.session_state.producer_started = True
            start_simulated_traffic(X_test=X_test) if X_test is not None else start_simulated_traffic()

        if modelo_clasificacion is not None and y_test is not None:
            show_architecture_2.show(modelo_clasificacion, y_test)

    # === DBSCAN ===
    elif st.session_state.seleccion == "dbscan":
        st.subheader("🧠 Clustering con DBSCAN")

        use_default = st.checkbox("Utilizar modelo DBSCAN predeterminado")
        modelo_dbscan = (
            default_dbsan_model() if use_default
            else load_model(label="Cargar modelo DBSCAN (.pkl)")
        )

        if modelo_dbscan is not None:
            X_tsne = load_X_tsne()
            y_train_class3, y_train_class2 = load_y_train_class2()
            if X_tsne is not None and y_train_class2 is not None:
                show_dbscan.show(modelo_dbscan, X_tsne, y_train_class3, y_train_class2)

    # === Isolation Forest ===
    elif st.session_state.seleccion == "isolation":
        st.subheader("🚨 Detección de Anomalías con Isolation Forest")

        use_default = st.checkbox("Utilizar modelo Isolation Forest predeterminado")
        modelo_isolation = (
            default_isolation_model() if use_default
            else load_model(label="Cargar modelo Isolation Forest (.pkl)")
        )

        if modelo_isolation is not None:
            X_tsne_train, X_tsne_test, y_train_class3_train, y_train_class3_test = load_X_tsne_con_test()
            if X_tsne_train is not None and X_tsne_test is not None and y_train_class3_test is not None:
                show_isolation.show(modelo_isolation, X_tsne_test, y_train_class3_test)
