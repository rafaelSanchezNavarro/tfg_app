import os
import streamlit as st
import pandas as pd
import time
from app.logic.evaluate_architecture_2 import evaluar_arquitectura_2_tiempo_real
from app.utils.anomalies import anomalies, encontrar_categoria
import warnings
import tempfile
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
# from app.logic.consumer import consume_traffic_from_kafka
from app.logic.consumer import consume_traffic_simulado


def inicializar_lime_explainer(modelo):
    if "lime_explainer" not in st.session_state:
        path_data = os.path.join('app', 'data')
        X_train = pd.read_csv(os.path.join(path_data, "X_train.csv"))

        st.session_state.X_train_lime = X_train
        st.session_state.lime_explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=list(modelo.classes_),
            mode='classification',
            random_state=42
        )

    return st.session_state.lime_explainer

def show(modelo, y_test):
    
    if "lime_explainer" not in st.session_state:
        st.session_state.lime_explainer = inicializar_lime_explainer(modelo)

    lime_explainer = st.session_state.lime_explainer
    
    if "real_time_running" not in st.session_state:
        st.session_state.real_time_running = False
    anomaly_detected = False
    
    if "explicaciones_container" not in st.session_state:
        st.session_state.explicaciones_container = st.empty()

    if st.button("▶️ Iniciar Simulación de Tráfico" if not st.session_state.real_time_running else "⏹️ Detener Simulación"):
        st.session_state.real_time_running = not st.session_state.real_time_running

    st.markdown("---")

    if st.session_state.real_time_running:
        st.subheader("🚦 Evaluación en Tiempo Real")
        tabla_container = st.empty()

        instancias_procesadas = 0
        anomalias_detectadas = []
        y_true_all = []
        y_pred_all = []

        while st.session_state.real_time_running:
            try:
                batch = next(consume_traffic_simulado())
                traffic_data_list = []
                indices = []

                for traffic_data, index in batch:
                    traffic_data_list.append(traffic_data)
                    indices.append(index)

                X_test_batch = pd.DataFrame(traffic_data_list)
                instancias_procesadas += len(X_test_batch)

                y_test_batch = pd.Series(y_test, index=range(len(y_test)))
                y_pred, y_test = evaluar_arquitectura_2_tiempo_real(modelo, X_test_batch, y_test_batch.loc[X_test_batch.index])

                y_true_all.extend(list(y_test))
                y_pred_all.extend(list(y_pred))

                anomalias_detectadas.extend([indices[i] for i, pred in enumerate(y_pred) if pred != 'Normal'])

                
                with tabla_container.container():
                    st.markdown("""
                        <style>
                            table.pred-table {
                                width: 100%;
                                border-collapse: collapse;
                                font-family: 'Segoe UI', sans-serif;
                                font-size: 12px;
                                margin-top: 10px;
                            }
                            .pred-table th, .pred-table td {
                                border: 1px solid #ccc;
                                padding: 6px 8px;
                                text-align: center;
                                vertical-align: middle;
                            }
                            .pred-table th {
                                background-color: #f0f0f0;
                                font-weight: bold;
                                color: #333;
                            }
                            .anomaly-row {
                                background-color: #ffecec;
                            }
                            .lime-table {
                                width: 100%;
                                font-size: 11px;
                                border-collapse: collapse;
                                margin-top: 4px;
                            }
                            .lime-table td {
                                padding: 2px 6px;
                            }
                        </style>
                    """, unsafe_allow_html=True)

                    # Cabecera de la tabla
                    st.markdown("""
                        <table class="pred-table">
                            <thead>
                                <tr>
                                    <th>Predicho</th>
                                    <th>Explicación</th>
                                </tr>
                            </thead>
                            <tbody>
                    """, unsafe_allow_html=True)

                    for idx, (pred, index) in enumerate(zip(y_pred, indices)):
                        color = "red" if pred != "Normal" else "black"
                        warning_emojis = "⚠️ " if pred != "Normal" else ""
                        closing_emojis = " ⚠️" if pred != "Normal" else ""


                        # Crear 3 columnas con celdas centradas, incluida la del expander
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown(
                                f"<div style='text-align: center; margin-left: -1.59cm; color: {color};'>{warning_emojis}{pred}{closing_emojis}</div>",
                                unsafe_allow_html=True
                            )
                        with col2:
                           expander_label = f"Normal" if pred == "Normal" else f"Anomalía"
                            with st.expander(expander_label):
                                with st.spinner(f'Generando explicación para instancia {index}...'):
                                    exp = lime_explainer.explain_instance(
                                        X_test_batch.iloc[idx],
                                        modelo.predict_proba,
                                        num_features=5
                                    )
                                    expl_html = "<table class='lime-table'>"
                                    for feature, weight in exp.as_list():
                                        color = "green" if weight >= 0 else "red"
                                        expl_html += f"<tr><td>{feature}</td><td style='color:{color}; text-align:right;'>{weight:.4f}</td></tr>"
                                    expl_html += "</table>"
                                    st.markdown(expl_html, unsafe_allow_html=True)

                    st.markdown("</tbody></table>", unsafe_allow_html=True)

                time.sleep(0.5)

            except StopIteration:
                st.warning("No hay más datos disponibles en el tópico.")
                break
            
