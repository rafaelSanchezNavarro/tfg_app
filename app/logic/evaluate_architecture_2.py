import numpy as np
import pandas as pd
from app.utils.anomalies import anomalies
import streamlit as st


def evaluar_arquitectura_2_tiempo_real(modelo, X_test, y_test_class1):
    # Si X_test es un diccionario, convértelo a DataFrame
    if isinstance(X_test, dict):
        X_test = pd.DataFrame({key: [value] for key, value in X_test.items()})
        
    y_test_class1 = y_test_class1.values.ravel()
    y_pred_class1_cat = modelo.predict(X_test)
    
    predicciones_df = pd.DataFrame({'Predicciones_Class1': y_pred_class1_cat})   
    
    return predicciones_df['Predicciones_Class1'], y_test_class1