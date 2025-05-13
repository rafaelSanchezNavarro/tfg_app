import joblib
from sklearn.base import BaseEstimator

def load_model(file):
    try:
        model = joblib.load(file)
        if not isinstance(model, BaseEstimator):
            raise ValueError("Archivo no contiene un modelo válido")
        return model
    except Exception as e:
        raise RuntimeError(f"Error cargando modelo: {str(e)}")