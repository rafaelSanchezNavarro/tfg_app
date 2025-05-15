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
    st.title("Trabajo de Fin de Grado - Aplicación de Análisis de Datos")

    st.markdown("""
    ### Evaluación de Modelos de Machine Learning

    Esta herramienta permite evaluar modelos de clasificación supervisada y clustering no supervisado. Puedes usar tus propios modelos o probar con modelos preentrenados.

    A continuación se presentan las distintas funcionalidades disponibles:
    """)

    with st.expander("🔍 Clasificación Supervisada"):
        st.markdown("""
        Evalúa modelos supervisados utilizando un flujo de tráfico simulado. Puedes cargar tu propio modelo o utilizar uno predeterminado. 
        El sistema muestra cómo el modelo detecta anomalías en tiempo real, incluyendo explicaciones breves de las decisiones del modelo.
        """)

    with st.expander("🧩 Clustering con DBSCAN"):
        st.markdown("""
        Aplica un modelo de clustering no supervisado con DBSCAN. Visualiza agrupaciones en los datos y consulta métricas que indican la capacidad del modelo 
        para distinguir entre comportamientos normales y anómalos.
        """)

    with st.expander("🧩 Detección de Anomalías con Isolation Forest"):
        st.markdown("""
        Usa un modelo Isolation Forest para identificar comportamientos anómalos. Permite comparar resultados con datos reales 
        y visualizar métricas que evalúan su rendimiento.
        """)

    st.markdown("Selecciona una opción para comenzar:")

    if "seleccion" not in st.session_state:
        st.session_state.seleccion = None

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔍 Clasificación"):
            st.session_state.seleccion = "clasificacion"
    with col2:
        if st.button("🧩 DBSCAN"):
            st.session_state.seleccion = "dbscan"
    with col3:
        if st.button("🧩 Isolation Forest"):
            st.session_state.seleccion = "isolation"

    st.markdown("---")

    # Clasificación Supervisada
    if st.session_state.seleccion == "clasificacion":
        st.subheader("🔍 Clasificación Supervisada")

        use_default_model = st.checkbox("Usar modelo predeterminado")
        use_default_test = st.checkbox("Usar conjunto de prueba predeterminado")

        # Cargar modelo
        if use_default_model:
            modelo_clasificacion = default_supervised_model()
        else:
            modelo_clasificacion = load_model(label="Cargar modelo de clasificación (.pkl)")

        # Cargar datos de prueba
        if use_default_test:
            X_test = None  # Se usará dentro de la función por defecto
            y_test = load_y_test()
        else:
            st.markdown("#### Cargar conjunto de prueba manual")
            X_test = load_csv(label="Cargar archivo de entrada X_test (.csv)")
            y_test = load_csv(label="Cargar archivo de etiquetas y_test (.csv)")
            
            # Convertir y_test a Series si viene como DataFrame de una sola columna
            if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
                y_test = y_test.iloc[:, 0]

        # Iniciar tráfico simulado una sola vez
        if "producer_started" not in st.session_state and X_test is not None:
            st.session_state.producer_started = True
            start_simulated_traffic(X_test=X_test)
        elif "producer_started" not in st.session_state and X_test is None:
            st.session_state.producer_started = True
            start_simulated_traffic()

        if modelo_clasificacion is not None and y_test is not None:
            show_architecture_2.show(modelo_clasificacion, y_test)



    # === DBSCAN ===
    elif st.session_state.seleccion == "dbscan":
        st.subheader("🧩 Clustering con DBSCAN")

        use_default = st.checkbox("Usar modelo DBSCAN predeterminado")
        if use_default:
            modelo_dbscan = default_dbsan_model()
        else:
            modelo_dbscan = load_model(label="Cargar modelo DBSCAN (.pkl)")

        if modelo_dbscan is not None:
            X_tsne = load_X_tsne()
            y_train_class3, y_train_class2 = load_y_train_class2()
            if X_tsne is not None and y_train_class2 is not None:
                show_dbscan.show(modelo_dbscan, X_tsne, y_train_class3, y_train_class2)

    # === Isolation Forest ===
    elif st.session_state.seleccion == "isolation":
        st.subheader("🧩 Detección de Anomalías con Isolation Forest")

        use_default = st.checkbox("Usar modelo Isolation Forest predeterminado")
        if use_default:
            modelo_isolation = default_isolation_model()
        else:
            modelo_isolation = load_model(label="Cargar modelo Isolation Forest (.pkl)")

        if modelo_isolation is not None:
            X_tsne_train, X_tsne_test, y_train_class3_train, y_train_class3_test = load_X_tsne_con_test()
            if X_tsne_train is not None and X_tsne_test is not None and y_train_class3_test is not None:
                show_isolation.show(modelo_isolation, X_tsne_test, y_train_class3_test)
