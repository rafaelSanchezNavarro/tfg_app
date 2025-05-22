import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def show(modelo, X_tsne, y_train_class3, y_train_class2):
    if not hasattr(modelo, "labels_"):
        st.error("El modelo cargado no parece ser un modelo DBSCAN entrenado.")
        return

    etiquetas = modelo.labels_

    st.markdown("<h3 style='text-align: center;'>Distribución real - Clusters predichos</h3>", unsafe_allow_html=True)

    # Columnas para mostrar en paralelo
    col1, col2 = st.columns(2)
    
    # 👉 Columna izquierda: distribución completa (referencia)
    with col1:
        
        # Paleta de colores para clases reales
        unique_classes = np.unique(y_train_class2)
        palette_real = sns.color_palette("tab10", len(unique_classes))
        class_to_color = {cls: palette_real[i] for i, cls in enumerate(unique_classes)}

        fig_ref, ax_ref = plt.subplots(figsize=(6, 6))

        # Puntos coloreados por clase real
        for cls in unique_classes:
            idx = y_train_class2 == cls
            ax_ref.scatter(
                X_tsne[idx, 0],
                X_tsne[idx, 1],
                color=class_to_color[cls],
                s=2,
                label=str(cls)
            )

        ax_ref.set_title("Categorías reales en t-SNE")
        ax_ref.set_xlabel("Componente 1")
        ax_ref.set_ylabel("Componente 2")
        ax_ref.grid(False)
        ax_ref.legend(loc='best', fontsize='x-small', markerscale=3)
        st.pyplot(fig_ref)

    # 👉 Columna derecha: animación progresiva de la predicción
    with col2:
        
        placeholder = st.empty()
        total_puntos = X_tsne.shape[0]
        paso = 5000  # Tamaño del lote para la animación

        for i in range(0, total_puntos + paso, paso):
            fig_pred, ax_pred = plt.subplots(figsize=(6, 6))
            current_labels = etiquetas[:i]
            current_points = X_tsne[:i]

            scatter = ax_pred.scatter(
                current_points[:, 0],
                current_points[:, 1],
                c=current_labels,
                cmap='tab10',
                s=1
            )

            # Crear leyenda personalizada
            unique_labels = np.unique(current_labels)
            handles = []
            for label in unique_labels:
                handles.append(plt.Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    label=f'Cluster {label}' if label != -1 else 'Ruido',
                    markerfacecolor=plt.cm.tab10(label % 10),
                    markersize=6
                ))
            ax_pred.legend(
                handles=handles,
                loc='best',
                fontsize='x-small',
            )

            ax_pred.set_title(f"Progreso: {min(i, total_puntos)} / {total_puntos}")
            ax_pred.set_xlabel("Componente 1")
            ax_pred.set_ylabel("Componente 2")

            placeholder.pyplot(fig_pred)
            plt.close()
            time.sleep(0.05)


    # Convertir a arrays por seguridad
    y_pred = np.array(etiquetas)
    y_true = np.array(y_train_class2)

    # Crear DataFrame conjunto
    df = pd.DataFrame({
        'cluster': y_pred,
        'real': y_true
    })

    # Agrupar por cluster y clase real
    tabla = df.groupby(['cluster', 'real']).size().unstack(fill_value=0)
    tabla_pct = tabla.div(tabla.sum(axis=1), axis=0)

    # st.markdown("#### 📊 Distribución de clases reales por clúster (conteo)")
    # st.dataframe(tabla)

    # st.markdown("#### 📈 Porcentaje de clases reales por clúster")
    # st.dataframe(tabla_pct.style.format("{:.1%}"))

    # st.markdown("#### 🧠 Interpretación basada en proporción de clase 'Normal'")
    # for cluster in tabla_pct.index:
    #     pct_normal = tabla_pct.loc[cluster].get("Normal", 0)
    #     tipo = "🛑 Anomalía (ataque)" if pct_normal < 0.5 else "✅ Normal"
    #     st.markdown(f"- Clúster `{cluster}`: {pct_normal:.1%} Normal → **{tipo}**")

    # st.markdown("#### 🔍 Interpretación basada en clase mayoritaria")
    # for cluster in tabla_pct.index:
    #     mayoritaria = tabla_pct.loc[cluster].idxmax()
    #     tipo = "🛑 Anomalía (ataque)" if mayoritaria != "Normal" else "✅ Normal"
    #     st.markdown(f"- Clúster `{cluster}`: clase mayoritaria **{mayoritaria}** → **{tipo}**")

    # Mapear clústeres a 0=Normal, 1=Ataque según la regla < 50% Normal
    clusters_anomalos = tabla_pct.index[tabla_pct["Normal"] < 0.5].tolist()
    y_pred_bin = np.array([1 if c in clusters_anomalos else 0 for c in y_pred])

    st.markdown("### 📋 Métricas de clasificación")

    accuracy = accuracy_score(y_train_class3, y_pred_bin)
    precision = precision_score(y_train_class3, y_pred_bin, zero_division=0)
    recall = recall_score(y_train_class3, y_pred_bin, zero_division=0)
    f1 = f1_score(y_train_class3, y_pred_bin, zero_division=0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")

    # Matriz de confusión
    st.markdown("### 🔍 Matriz de confusión")
    labels = sorted(set(y_train_class3))
    label_names = [str(label) for label in labels]  
    cm = confusion_matrix(y_train_class3, y_pred_bin)
    cm_df = pd.DataFrame(cm, index=[f'Real {l}' for l in label_names], columns=[f'Pred {l}' for l in label_names])
    st.dataframe(cm_df, use_container_width=True)
