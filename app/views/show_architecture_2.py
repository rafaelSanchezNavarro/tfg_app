import os
import streamlit as st
import pandas as pd
import time
from app.logic.evaluate_architecture_2 import evaluar_arquitectura_2_tiempo_real
from app.utils.anomalies import anomalies, encontrar_categoria
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import matplotlib.pyplot as plt
import seaborn as sns
from app.logic.consumer import consume_traffic_from_kafka


def highlight(row):
    return ['background-color: red' if row['Predicho'] != 'Normal' else '' for _ in row.index]


def show(modelo, y_test):
    if "real_time_running" not in st.session_state:
        st.session_state.real_time_running = False
    anomaly_detected = False

    if st.button("▶️ Iniciar Simulación de Tráfico" if not st.session_state.real_time_running else "⏹️ Detener Simulación"):
        st.session_state.real_time_running = not st.session_state.real_time_running

    st.markdown("---")

    if st.session_state.real_time_running:
        st.subheader("🚦 Evaluación en Tiempo Real")
        result_container = st.empty()

        instancias_procesadas = 0
        anomalias_detectadas = []
        while st.session_state.real_time_running:
            try:
                batch = next(consume_traffic_from_kafka())
                traffic_data_list = []
                indices = []

                for traffic_data, index in batch:
                    traffic_data_list.append(traffic_data)
                    indices.append(index)

                X_test_batch = pd.DataFrame(traffic_data_list)
                instancias_procesadas += len(X_test_batch)

                y_test_batch = pd.Series(y_test, index=range(len(y_test)))
                y_pred, y_test = evaluar_arquitectura_2_tiempo_real(modelo, X_test_batch, y_test_batch.loc[X_test_batch.index])


                result_data = []
                indices_anomalias = []
                anomaly_detected = False

                for idx, (real, pred) in enumerate(zip(y_test, y_pred)):
                    if pred != 'Normal':
                        coincide = 'Anomalía detectada'
                        anomaly_detected = True
                        indices_anomalias.append(idx)
                    else:
                        coincide = ''

                    result_data.append({
                        "Predicho": pred,
                        "Alerta": coincide
                    })

                df = pd.DataFrame(result_data)
                anomalias_detectadas.extend([indices[i] for i in indices_anomalias])

                try:
                    for alert_container in alert_containers:
                        alert_container.empty()
                except:
                    pass

                styled_df = df.style.apply(highlight, axis=1)
                result_container.dataframe(styled_df, use_container_width=True, height=600)

                if anomaly_detected:
                    alert_containers = []
                    for idx in indices_anomalias:
                        alert_container = st.empty()
                        alert_containers.append(alert_container)
                        alert_container.warning(f"Anomalía detectada en la fila {idx}.", icon="⚠️")
                        # imprimir_explicacion(idx, y_pred, X_test_batch, modelo)

            except StopIteration:
                print("No more data available in the topic.")
                break
