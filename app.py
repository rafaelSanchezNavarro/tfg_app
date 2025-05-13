import streamlit as st
from app.views import inicio

def main():
    st.set_page_config(page_title="TFG", layout="wide")
    
    # Cargar CSS personalizado
    with open("app/static/css/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Mostrar directamente la vista de Evaluar Modelo
    inicio.show()


if __name__ == "__main__":
    main()
