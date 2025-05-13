import pandas as pd

def load_csv(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        raise RuntimeError(f"Error cargando CSV: {str(e)}")

def validate_data(X, y):
    if X.shape[0] != y.shape[0]:
        raise ValueError("Dimensión de datos y etiquetas no coinciden")