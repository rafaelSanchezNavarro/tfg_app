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
# from app.logic.consumer import consume_traffic_from_kafka
from app.logic.consumer import consume_traffic_simulado


def highlight(row):
    return ['background-color: red' if row['Predicho'] != 'Normal' else '' for _ in row.index]


def show(modelo, y_test):
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
        result_container = st.empty()

        instancias_procesadas = 0
        anomalias_detectadas = []
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

                styled_df = df.style.apply(highlight, axis=1)
                result_container.dataframe(styled_df, use_container_width=True, height=212)

                # if anomaly_detected:
                #     # Limpiar y generar nuevos botones dentro del contenedor
                #     with st.session_state.explicaciones_container.container():
                #         st.subheader("🔍 Explicaciones disponibles")
                #         for idx in indices_anomalias:
                #             url = f"https://example.com/explicacion?fila={idx}"  # URL ficticia, cámbiala según necesidad

                #             st.markdown(
                #                 f"""
                #                 <a href="{url}" target="_blank">
                #                     <button style="margin: 5px 0;">Ver explicación para fila {idx}</button>
                #                 </a>
                #                 """,
                #                 unsafe_allow_html=True
                #             )
                # else:
                #     # Si no hay anomalías, limpiar el contenedor
                #     st.session_state.explicaciones_container.empty()

                time.sleep(0.5)
            except StopIteration:
                print("No more data available in the topic.")
                break
