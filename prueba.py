import numpy as np
import streamlit as st
from app.logic import model_loading, data_processing, producer
import pandas as pd
import time
import zipfile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.utils.anomalies import anomalies
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

def evaluar_arquitectura_1_tiempo_real(modelos, X_test, y_tests):
    y_test_class3, y_test_class2, y_test_class1 = y_tests
    y_pred_class3 = modelos['capa1'].predict(X_test)
    # Asignar las predicciones de la capa 1 al DataFrame
    predicciones_df = pd.DataFrame({'Predicciones_Class3': y_pred_class3})  
    
    # Predicciones de la capa 2
    indices_anomalia_test = np.where(y_pred_class3 == 1)[0]
    indices_normal_test = np.where(y_pred_class3 == 0)[0]
    if 1 in y_pred_class3:
        X_test = X_test.iloc[indices_anomalia_test]
        y_test_class2 = y_test_class2.iloc[indices_anomalia_test].values.ravel()
        y_pred_class2 = modelos['capa2'].predict(X_test)
        # Asignar las predicciones de la capa 2 al DataFrame
        predicciones_df['Predicciones_Class2'] = "Normal"  
        predicciones_df.loc[indices_anomalia_test, 'Predicciones_Class2'] = y_pred_class2
        indices_normal_test = np.where(y_pred_class3 == 0)[0]
        predicciones_df.loc[indices_normal_test, 'Predicciones_Class2'] = "Normal" 
    else:
        predicciones_df['Predicciones_Class2'] = "Normal"  
        predicciones_df.loc[indices_normal_test, 'Predicciones_Class2'] = "Normal"
    
    # Predicciones de la capa 3
    y_test_class1 = y_test_class1.iloc[indices_anomalia_test].values.ravel()
    y_pred_class1_total = ["Normal"] * len(indices_anomalia_test)  
    categorias_multiples_tipos = [key for key, value in anomalies.items() if len(value) > 1]
    if 1 in y_pred_class3:
        for categoria in y_pred_class2:
            if categoria in categorias_multiples_tipos:

                indices_anomalia_prediccion_categoria = np.where(categoria)[0]
                X_test_categoria = X_test.iloc[indices_anomalia_prediccion_categoria]
                y_test_class1_categoria = y_test_class1[indices_anomalia_prediccion_categoria]
                
                model = modelos[categoria]
                
                y_pred_class1_categoria = model.predict(X_test_categoria)
                
                # Asignar las predicciones a la lista total usando el índice correcto
                for idx, pred in zip(indices_anomalia_prediccion_categoria, y_pred_class1_categoria):
                    y_pred_class1_total[idx] = pred
                    
                predicciones_df.loc[indices_anomalia_test, 'Predicciones_Class1'] = y_pred_class1_total 

    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "RDOS"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "RDOS"
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "Exfiltration"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "Exfiltration"
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "C&C"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "C&C"
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "crypto-ransomware"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "crypto-ransomware"
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "Normal"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "Normal"

    if len (predicciones_df['Predicciones_Class1']) == 0:
        return predicciones_df['Predicciones_Class2'], y_tests[2]
    return predicciones_df['Predicciones_Class1'], y_tests[2]

def logica_arquitectura_1_tiempo_real(y_test, y_pred):
    result_container = st.empty()

    df = pd.DataFrame({
        "Real": y_test.reset_index(drop=True),
        "Predicho": y_pred.reset_index(drop=True),
        "Coincide": ["✅" if r == p else "❌" for r, p in zip(y_test, y_pred)]
    })

    result_container.dataframe(df, use_container_width=True)

def evaluar_arquitectura_1(modelos, X_test, y_tests):
 
    # st.subheader("📈 Evaluación - Arquitectura 1")
    # Predicciones de la capa 1
    y_test_class3, y_test_class2, y_test_class1 = y_tests
    y_pred_class3 = modelos['capa1'].predict(X_test)
    accuracy = accuracy_score(y_test_class3, y_pred_class3)
    print(f"📈 Accuracy (Test): {accuracy:.4f}")
    precision = precision_score(y_test_class3, y_pred_class3, average='binary')
    print(f"📈 Precision (Test): {precision:.4f}")
    recall = recall_score(y_test_class3, y_pred_class3, average='binary')
    print(f"📈 Recall (Test): {recall:.4f}")
    f1 = f1_score(y_test_class3, y_pred_class3, average='binary')
    print(f"📈 F1 (Test): {f1:.4f}")
    # Asignar las predicciones de la capa 1 al DataFrame
    predicciones_df = pd.DataFrame({'Predicciones_Class3': y_pred_class3})  
    
    # Predicciones de la capa 2
    indices_anomalia_test = np.where(y_pred_class3 == 1)[0]
    X_test = X_test.iloc[indices_anomalia_test]
    y_test_class2 = y_test_class2.iloc[indices_anomalia_test].values.ravel()
    y_pred_class2 = modelos['capa2'].predict(X_test)
    indices_anomalias_reales = np.where(np.isin(y_test_class2, list(anomalies.keys())))[0]
    y_test_class2_real = y_test_class2[indices_anomalias_reales]
    y_pred_class2_real = y_pred_class2[indices_anomalias_reales]
    accuracy = accuracy_score(y_test_class2_real, y_pred_class2_real)
    print(f'📈 Accuracy (Test): {accuracy:.4f}')
    precision = precision_score(y_test_class2_real, y_pred_class2_real, average='macro')
    print(f'📈 Precision (Test): {precision:.4f}')
    recall = recall_score(y_test_class2_real, y_pred_class2_real, average='macro')
    print(f'📈 Recall (Test): {recall:.4f}')
    f1 = f1_score(y_test_class2_real, y_pred_class2_real, average='macro')
    print(f'📈 F1 (Test): {f1:.4f}')
    # Asignar las predicciones de la capa 2 al DataFrame
    predicciones_df['Predicciones_Class2'] = "Normal"  
    predicciones_df.loc[indices_anomalia_test, 'Predicciones_Class2'] = y_pred_class2
    
    
    # Predicciones de la capa 3
    y_test_class1 = y_test_class1.iloc[indices_anomalia_test].values.ravel()
    y_pred_class1_total = ["Normal"] * len(indices_anomalia_test)  
    categorias_multiples_tipos = [key for key, value in anomalies.items() if len(value) > 1]
    
    for categoria in categorias_multiples_tipos:
        
        indices_anomalia_prediccion_categoria = np.where(y_pred_class2 == categoria)[0]
        X_test_categoria = X_test.iloc[indices_anomalia_prediccion_categoria]
        y_test_class1_categoria = y_test_class1[indices_anomalia_prediccion_categoria]
        
        model = modelos[categoria]
        
        y_pred_class1_categoria = model.predict(X_test_categoria)
        
        # Asignar las predicciones a la lista total usando el índice correcto
        for idx, pred in zip(indices_anomalia_prediccion_categoria, y_pred_class1_categoria):
            y_pred_class1_total[idx] = pred
        
        indices_anomalias_reales = np.where(np.isin(y_test_class1_categoria, anomalies.get(categoria, [])))[0]
        y_test_class1_real = y_test_class1_categoria[indices_anomalias_reales]
        y_pred_class1_real = y_pred_class1_categoria[indices_anomalias_reales]

        accuracy = accuracy_score(y_test_class1_real, y_pred_class1_real)
        print(f'📈 Accuracy: {accuracy:.4f}')
        precision = precision_score(y_test_class1_real, y_pred_class1_real, average='macro', zero_division=0)
        print(f'📈 Precision: {precision:.4f}')
        recall = recall_score(y_test_class1_real, y_pred_class1_real, average='macro')
        print(f'📈 Recall: {recall:.4f}')
        f1 = f1_score(y_test_class1_real, y_pred_class1_real, average='macro')
        print(f'📈 F1: {f1:.4f}')      
        
        predicciones_df.loc[indices_anomalia_test, 'Predicciones_Class1'] = y_pred_class1_total 
        
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "RDOS"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "RDOS"
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "Exfiltration"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "Exfiltration"
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "C&C"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "C&C"
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "crypto-ransomware"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "crypto-ransomware"
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "Normal"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "Normal"
    
    logica_arquitectura_1(y_tests[2], predicciones_df['Predicciones_Class1'])
    
    return predicciones_df['Predicciones_Class1'], y_tests[2]

def evaluar_arquitectura_2(modelo, X_test, y_test):
    st.subheader("📈 Evaluación - Arquitectura 2")
    y_pred = modelo.predict(X_test)
    logica_arquitectura_2(y_test, y_pred)
    return y_pred

def evaluar_arquitectura_3(modelos, X_test, y_tests):
    st.subheader("📈 Evaluación - Arquitectura 3")
    
    y_test_class2, y_test_class1 = y_tests
    y_test_class2 = y_test_class2.values.ravel()
    y_pred_class2 = modelos['capa1'].predict(X_test)
    accuracy = accuracy_score(y_test_class2, y_pred_class2)
    print(f'📈 Accuracy (Test): {accuracy:.4f}')
    precision = precision_score(y_test_class2, y_pred_class2, average='macro', zero_division=0)
    print(f'📈 Precision (Test): {precision:.4f}')
    recall = recall_score(y_test_class2, y_pred_class2, average='macro')
    print(f'📈 Recall (Test): {recall:.4f}')
    f1 = f1_score(y_test_class2, y_pred_class2, average='macro')
    print(f'📈 F1 (Test): {f1:.4f}')
    predicciones_df = pd.DataFrame({'Predicciones_Class2': y_pred_class2}) 
    
    
    indices_anomalia_test = np.where(y_pred_class2 != "Normal")[0]
    X_test = X_test.iloc[indices_anomalia_test]
    y_pred_class2 = y_pred_class2[indices_anomalia_test]
    y_test_class1 = y_test_class1.iloc[indices_anomalia_test].values.ravel()
    

    y_pred_class1_total = ["Normal"] * len(indices_anomalia_test)  # Inicializa con "Normal" 
    
    categorias_multiples_tipos = [key for key, value in anomalies.items() if len(value) > 1]
    for categoria in categorias_multiples_tipos:
        
        print(f"🔮 Clasificación multiclase (Tipo) para {categoria}...")
        indices_anomalia_prediccion_categoria = np.where(y_pred_class2 == categoria)[0]
        X_test_categoria = X_test.iloc[indices_anomalia_prediccion_categoria]
        y_test_class1_categoria = y_test_class1[indices_anomalia_prediccion_categoria]
        
        model = modelos[categoria]

        y_pred_class1_categoria = model.predict(X_test_categoria)
        
        # Asignar las predicciones a la lista total usando el índice correcto
        for idx, pred in zip(indices_anomalia_prediccion_categoria, y_pred_class1_categoria):
            y_pred_class1_total[idx] = pred
        
        indices_anomalias_reales = np.where(np.isin(y_test_class1_categoria, anomalies.get(categoria, [])))[0]
        y_test_class1_real = y_test_class1_categoria[indices_anomalias_reales]
        y_pred_class1_real = y_pred_class1_categoria[indices_anomalias_reales]

        accuracy = accuracy_score(y_test_class1_real, y_pred_class1_real)
        print(f'📈 Accuracy: {accuracy:.4f}')
        precision = precision_score(y_test_class1_real, y_pred_class1_real, average='macro', zero_division=0)
        print(f'📈 Precision: {precision:.4f}')
        recall = recall_score(y_test_class1_real, y_pred_class1_real, average='macro')
        print(f'📈 Recall: {recall:.4f}')
        f1 = f1_score(y_test_class1_real, y_pred_class1_real, average='macro')
        print(f'📈 F1: {f1:.4f}')    
        
        predicciones_df.loc[indices_anomalia_test, 'Predicciones_Class1'] = y_pred_class1_total 
        
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "RDOS"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "RDOS"
    
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "Exfiltration"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "Exfiltration"
    
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "C&C"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "C&C"
    
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "crypto-ransomware"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "crypto-ransomware"
    
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "Normal"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "Normal"
    
    logica_arquitectura_3(y_tests[1], predicciones_df['Predicciones_Class1'])
    
    return predicciones_df['Predicciones_Class1']

def logica_arquitectura_1(y_test, y_pred):
    df = pd.DataFrame({
        "Real": y_test,
        "Predicho": y_pred,
        "Coincide": ["✅" if r == p else "❌" for r, p in zip(y_test, y_pred)]
    })
    st.dataframe(df, use_container_width=True, height=600)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    st.markdown("### 📐 Métricas de Evaluación")
    st.metric("Accuracy", f"{acc:.4%}")
    st.metric("Precisión", f"{prec:.4%}")
    st.metric("Recall", f"{rec:.4%}")
    st.metric("F1 Score", f"{f1:.4%}")

def logica_arquitectura_2(y_test, y_pred):
    df = pd.DataFrame({
        "Real": y_test,
        "Predicho": y_pred,
        "Coincide": ["✅" if r == p else "❌" for r, p in zip(y_test, y_pred)]
    })
    st.dataframe(df, use_container_width=True)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    st.markdown("### 📐 Métricas de Evaluación")
    st.metric("Accuracy", f"{acc:.2%}")
    st.metric("Precisión", f"{prec:.2%}")
    st.metric("Recall", f"{rec:.2%}")
    st.metric("F1 Score", f"{f1:.2%}")

def logica_arquitectura_3(y_test, y_pred):
    df = pd.DataFrame({
        "Real": y_test,
        "Predicho": y_pred,
        "Coincide": ["✅" if r == p else "❌" for r, p in zip(y_test, y_pred)]
    })
    st.dataframe(df, use_container_width=True)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    st.markdown("### 📐 Métricas de Evaluación")
    st.metric("Accuracy", f"{acc:.2%}")
    st.metric("Precisión", f"{prec:.2%}")
    st.metric("Recall", f"{rec:.2%}")
    st.metric("F1 Score", f"{f1:.2%}")
    
def show():
    st.title("📊 Evaluación de Modelos de ML")

    if "real_time_running" not in st.session_state:
        st.session_state.real_time_running = False
    if "full_eval_running" not in st.session_state:
        st.session_state.full_eval_running = False

    with st.container():
        st.header("⚙️ Configuración")
        arquitectura = st.selectbox("Seleccione Arquitectura", ["Arquitectura 1", "Arquitectura 2", "Arquitectura 3"])

        if arquitectura == "Arquitectura 1":
            st.markdown("**📅 Modelos por Capas**")
            capa1_model = st.file_uploader("Capa 1 - Modelo (.pkl)", type="pkl", key="c1a1")
            capa2_model = st.file_uploader("Capa 2 - Modelo (.pkl)", type="pkl", key="c2a1")
            capa3_zip = st.file_uploader("Capa 3 - ZIP de Modelos", type="zip", key="c3a1")
            X_test = st.file_uploader("📄 Subir X_test.csv", type="csv", key="xa1")
            y_test_class1 = st.file_uploader("📄 y_test_class1.csv", type="csv")
            y_test_class2 = st.file_uploader("📄 y_test_class2.csv", type="csv")
            y_test_class3 = st.file_uploader("📄 y_test_class3.csv", type="csv")
        elif arquitectura == "Arquitectura 3":
            st.markdown("**📅 Modelos por Capas**")
            capa1_model = st.file_uploader("Capa 1 - Modelo (.pkl)", type="pkl", key="c1a3")
            capa2_zip = st.file_uploader("Capa 2 - ZIP de Modelos", type="zip", key="c2a3")
            X_test = st.file_uploader("📄 Subir X_test.csv", type="csv")
            y_test_class1 = st.file_uploader("📄 y_test_class1.csv", type="csv")
            y_test_class2 = st.file_uploader("📄 y_test_class2.csv", type="csv")
        else:
            modelo = st.file_uploader("🔄 Subir Modelo (.pkl)", type="pkl")
            X_test = st.file_uploader("📄 Subir X_test.csv", type="csv")
            y_file = st.file_uploader("📄 Subir y_test.csv", type="csv")

    modelos_cargados = {}
    y_tests = []

    try:
        if arquitectura == "Arquitectura 1":
            if capa1_model and capa2_model and capa3_zip and X_test and y_test_class1 and y_test_class2 and y_test_class3:
                modelos_cargados["capa1"] = model_loading.load_model(capa1_model)
                modelos_cargados["capa2"] = model_loading.load_model(capa2_model)

                with zipfile.ZipFile(capa3_zip, "r") as zip_ref:
                    for name in zip_ref.namelist():
                        
                        for categoria in anomalies.keys():
                            if f"{categoria}_class1.pkl" == name:
                                with zip_ref.open(name) as f:
                                    modelos_cargados[categoria] = model_loading.load_model(f)

                
                X_test = data_processing.load_csv(X_test)
                y_test_class1 = data_processing.load_csv(y_test_class1).squeeze()
                y_test_class2 = data_processing.load_csv(y_test_class2).squeeze()
                y_test_class3 = data_processing.load_csv(y_test_class3).squeeze()
                y_tests = [y_test_class3, y_test_class2, y_test_class1]
                
        elif arquitectura == "Arquitectura 2":
            if modelo and X_test and y_file:
                modelo = model_loading.load_model(modelo)
                X_test = data_processing.load_csv(X_test)
                y_test = data_processing.load_csv(y_file).squeeze()
                data_processing.validate_data(X_test, y_test)
                
        elif arquitectura == "Arquitectura 3":
            if capa1_model and  capa2_zip and X_test and y_test_class1 and y_test_class2:
                modelos_cargados["capa1"] = model_loading.load_model(capa1_model)

                with zipfile.ZipFile(capa2_zip, "r") as zip_ref:
                    for name in zip_ref.namelist():
                        
                        for categoria in anomalies.keys():
                            if f"{categoria}_class1.pkl" == name:
                                with zip_ref.open(name) as f:
                                    modelos_cargados[categoria] = model_loading.load_model(f)

                X_test = data_processing.load_csv(X_test)
                y_test_class1 = data_processing.load_csv(y_test_class1).squeeze()
                y_test_class2 = data_processing.load_csv(y_test_class2).squeeze()
                y_tests = [y_test_class2, y_test_class1]

        if arquitectura == "Arquitectura 1" and modelos_cargados.get("capa1") and modelos_cargados.get("capa2") and X_test is not None and all(len(y) > 0 for y in y_tests):
            st.success("✅ Archivos cargados correctamente")
            st.markdown("---")

        elif arquitectura == "Arquitectura 2" and modelo and X_test is not None and y_test is not None and len(y_test) > 0:
            st.success("✅ Archivos cargados correctamente")
            st.markdown("---")

        elif arquitectura == "Arquitectura 3" and modelos_cargados.get("capa1") and X_test is not None and all(len(y) > 0 for y in y_tests):
            st.success("✅ Archivos cargados correctamente")
            st.markdown("---")


        col1, col2 = st.columns(2)

        if col1.button("▶️ Iniciar Simulación de Tráfico" if not st.session_state.real_time_running else "⏹️ Detener Simulación"):
            st.session_state.real_time_running = not st.session_state.real_time_running
            st.session_state.full_eval_running = False

        if col2.button("▶️ Iniciar Evaluación Completa" if not st.session_state.full_eval_running else "⏹️ Detener Evaluación"):
            st.session_state.full_eval_running = not st.session_state.full_eval_running
            st.session_state.real_time_running = False

        st.markdown("---")

        if st.session_state.real_time_running:
            st.subheader("🚦 Evaluación en Tiempo Real")
            result_container = st.empty()

            while st.session_state.real_time_running:
                traffic_data = producer.simulate_traffic(X_test, 16)
                
                if arquitectura == "Arquitectura 1":
                    # Obtener los valores de y_test para cada clase y alinearlos con traffic_data
                    y_tests = [y_test_class3.loc[traffic_data.index], 
                            y_test_class2.loc[traffic_data.index], 
                            y_test_class1.loc[traffic_data.index]]
                    y_pred, y_test = evaluar_arquitectura_1_tiempo_real(modelos_cargados, traffic_data, y_tests)
                    
                elif arquitectura == "Arquitectura 2":
                    y_pred = evaluar_arquitectura_2(modelo, traffic_data, y_test)
                elif arquitectura == "Arquitectura 3":
                    y_pred = evaluar_arquitectura_3(modelos_cargados, traffic_data, y_tests)
                
                result_data = []

                for real, pred in zip(y_test, y_pred):
                    result_data.append({
                        "Real": real,
                        "Predicho": pred,
                        "Coincide": "✅" if real == pred else "❌"
                    })

                df = pd.DataFrame(result_data)
                result_container.dataframe(df, use_container_width=True, height=600)
                time.sleep(2)


        if st.session_state.full_eval_running:
            if arquitectura == "Arquitectura 1":
                evaluar_arquitectura_1(modelos_cargados, X_test, y_tests)
            elif arquitectura == "Arquitectura 2":
                evaluar_arquitectura_2(modelo, X_test, y_test)
            elif arquitectura == "Arquitectura 3":
                evaluar_arquitectura_3(modelos_cargados, X_test, y_tests)
                

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
