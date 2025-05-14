import streamlit as st
from app.logic.load_data import (
    load_X_tsne_con_test,
    load_model,
    load_y_test,
    load_X_tsne,
    load_y_train_class2,
    default_supervised_model,
    default_isolation_model,
    default_dbsan_model,
)
from app.logic.producer import start_kafka_producer
from app.views import show_architecture_2, show_dbscan, show_isolation

def show():
    st.title("Trabajo de Fin de Grado - Aplicación de Análisis de Datos")

    st.markdown("""
    ### Evaluación de Modelos de Machine Learning

    Esta herramienta permite evaluar modelos de clasificación supervisada y clustering no supervisado. Puedes usar tus propios modelos o probar con modelos preentrenados.

    **Opciones disponibles:**
    - Clasificación Supervisada
    - Clustering con DBSCAN
    - Detección de Anomalías con Isolation Forest
    """)

    if "seleccion" not in st.session_state:
        st.session_state.seleccion = None

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔍 Clasificación Supervisada"):
            st.session_state.seleccion = "clasificacion"
    with col2:
        if st.button("🧩 Clustering DBSCAN"):
            st.session_state.seleccion = "dbscan"
    with col3:
        if st.button("🧩 Isolation Forest"):
            st.session_state.seleccion = "isolation"

    st.markdown("---")

    # === Clasificación Supervisada ===
    if st.session_state.seleccion == "clasificacion":
        st.subheader("🔍 Clasificación Supervisada")

        # Lanzar el productor solo una vez
        if "producer_started" not in st.session_state:
            st.session_state.producer_started = True
            start_kafka_producer()

        use_default = st.checkbox("Usar modelo predeterminado")
        if use_default:
            modelo_clasificacion = default_supervised_model()
        else:
            modelo_clasificacion = load_model(label="Cargar modelo de clasificación (.pkl)")

        if modelo_clasificacion is not None:
            y_test = load_y_test()
            if y_test is not None:
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