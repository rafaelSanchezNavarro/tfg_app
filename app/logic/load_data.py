import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.utils import resample

def load_model(label="Selecciona un modelo (.pkl)"):
    with st.container():
        modelo_file = st.file_uploader(label, type="pkl")

    modelo = None
    if modelo_file is not None:
        try:
            modelo = joblib.load(modelo_file)
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {e}")
    return modelo


def default_supervised_model():
    try:
        path = os.path.join("app", "data", "DecisionTreeClassifier_0.9940.pkl")
        modelo = joblib.load(path)
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo supervisado: {e}")
        return None
    
def default_dbsan_model():
    try:
        path = os.path.join("app", "data", "dbscan_model.pkl")
        modelo = joblib.load(path)
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo no supervisado: {e}")
        return None
    
def default_isolation_model():
    try:
        path = os.path.join("app", "data", "iso_forest_todo.pkl")
        modelo = joblib.load(path)
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo no supervisado: {e}")
        return None
    
def load_y_test():
    try:
        path = os.path.join("app", "data", "y_test_class1_filtrado.csv")
        y_test = pd.read_csv(path, low_memory=False).squeeze()
        return y_test
    except Exception as e:
        st.error(f"❌ Error al cargar y_test: {e}")
        return None


def load_X_tsne():
    try:
        path = os.path.join("app", "data", "X_tsne.npy")
        X_tsne = np.load(path)
        return X_tsne
    except Exception as e:
        st.error(f"❌ Error al cargar X_tsne.npy: {e}")
        return None


def load_X_tsne_con_test():
    try:
        path = os.path.join("app", "data", "X_tsne.npy")
        X_tsne = np.load(path)
        
        path = os.path.join("app", "data", "y_train_class3.csv")
        y_train_class3 = pd.read_csv(path, low_memory=False).squeeze()
        
        y_train_class3= resample(
            y_train_class3,
            n_samples=100000,
            replace=False,
            random_state=42
        )
        
        X_tsne_train, X_tsne_test, y_train_class3_train, y_train_class3_test = train_test_split(
            X_tsne, y_train_class3, test_size=0.3, random_state=42
        )
                
        return X_tsne_train, X_tsne_test, y_train_class3_train, y_train_class3_test
    except Exception as e:
        st.error(f"❌ Error al cargar X_tsne.npy: {e}")
        return None

def load_y_train_class2():
    try:
    
        path = os.path.join("app", "data", "y_train_class2.csv")
        y_train_class2 = pd.read_csv(path, low_memory=False).squeeze()
        
        path = os.path.join("app", "data", "y_train_class3.csv")
        y_train_class3 = pd.read_csv(path, low_memory=False).squeeze()
        
        y_train_class3, y_train_class2= resample(
            y_train_class3, y_train_class2, 
            n_samples=100000,
            replace=False,
            random_state=42
        )
        
        return y_train_class3, y_train_class2
    
    except Exception as e:
        st.error(f"❌ Error al cargar y_train_class2: {e}")
        return None
