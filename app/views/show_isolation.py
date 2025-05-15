import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def show(modelo_isolation, X_tsne_test, y_train_class3_test):
    pred_test = modelo_isolation.predict(X_tsne_test)
    pred_test = np.where(pred_test == -1, 1, 0)  # 1 = anomalía, 0 = normal

    # Extraer clases reales
    unique_classes = np.unique(y_train_class3_test)
    palette_real = sns.color_palette("tab10", len(unique_classes))
    class_to_color = {cls: palette_real[i] for i, cls in enumerate(unique_classes)}

    # Mapear predicciones a clases reales
    clase_normal = next((cls for cls in unique_classes if "normal" in str(cls).lower()), unique_classes[0])
    clase_anomalo = next((cls for cls in unique_classes if cls != clase_normal), unique_classes[-1])

    color_normal = class_to_color[clase_normal]
    color_anomalo = class_to_color[clase_anomalo]

    st.markdown("<h3 style='text-align: center;'>Comportamiento real - Comportamiento predicho</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    # 👉 Columna izquierda: clases reales
    with col1:
        
        fig_real, ax_real = plt.subplots(figsize=(6, 6))
        for cls in unique_classes:
            idx = y_train_class3_test == cls
            label = None
            if cls == clase_normal:
                label = "Normal"
            elif cls == clase_anomalo:
                label = "Anomalo"

            ax_real.scatter(
                X_tsne_test[idx, 0],
                X_tsne_test[idx, 1],
                color=class_to_color[cls],
                s=2,
                label=label  # Solo etiqueta si es Normal o Anomalía
            )

        ax_real.set_title("Comportamiento real en t-SNE")
        ax_real.set_xlabel("Componente 1")
        ax_real.set_ylabel("Componente 2")
        ax_real.grid(False)
        ax_real.legend(loc='best', fontsize='small', markerscale=3)
        st.pyplot(fig_real)

    # 👉 Columna derecha: predicción progresiva
    with col2:
        
        placeholder = st.empty()
        total_puntos = X_tsne_test.shape[0]
        paso = 3000 # Tamaño del lote para la animación

        for i in range(0, total_puntos + paso, paso):
            fig_pred, ax_pred = plt.subplots(figsize=(6, 6))

            current_points = X_tsne_test[:i]
            current_preds = pred_test[:i]
            colors = np.array([color_normal if p == 0 else color_anomalo for p in current_preds])

            ax_pred.scatter(
                current_points[:, 0],
                current_points[:, 1],
                color=colors,
                s=2
            )

            # Leyenda personalizada
            leyenda = [
                Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor=color_normal, markersize=6),
                Line2D([0], [0], marker='o', color='w', label='Anomalo', markerfacecolor=color_anomalo, markersize=6)
            ]

            ax_pred.legend(handles=leyenda, loc='best', fontsize='x-small')

            ax_pred.set_title(f"Progreso: {min(i, total_puntos)} / {total_puntos}")
            ax_pred.set_xlabel("Componente 1")
            ax_pred.set_ylabel("Componente 2")
            ax_pred.grid(False)

            placeholder.pyplot(fig_pred)
            plt.close()
            time.sleep(0.05)

    # Calcular métricas
    accuracy = accuracy_score(y_train_class3_test, pred_test)
    precision = precision_score(y_train_class3_test, pred_test, average='binary', zero_division=0)
    recall = recall_score(y_train_class3_test, pred_test, average='binary', zero_division=0)
    f1 = f1_score(y_train_class3_test, pred_test, average='binary')

    # Mostrar métricas con Streamlit
    st.markdown("### 📊 Métricas de evaluación (conjunto de test)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Accuracy", f"{accuracy:.2%}")
    col2.metric("📌 Precision", f"{precision:.2%}")
    col3.metric("📈 Recall", f"{recall:.2%}")
    col4.metric("📊 F1 Score", f"{f1:.2%}")

    # Matriz de confusión
    st.markdown("### 🔍 Matriz de confusión")
    labels = sorted(set(y_train_class3_test))
    label_names = [str(label) for label in labels]  
    cm = confusion_matrix(y_train_class3_test, pred_test)
    cm_df = pd.DataFrame(cm, index=[f'Real {l}' for l in label_names], columns=[f'Pred {l}' for l in label_names])
    st.dataframe(cm_df, use_container_width=True)