# 🎓 Trabajo de Fin de Grado – Análisis de Modelos de Machine Learning

Este proyecto es una aplicación desarrollada en Python y Streamlit para evaluar modelos de Machine Learning aplicados a tráfico de red. La aplicación permite analizar modelos supervisados y no supervisados mediante una interfaz visual interactiva. También simula tráfico en tiempo real utilizando Kafka.

---

## 🧠 Funcionalidades principales

- 🔍 **Clasificación Supervisada**  
  Carga o usa un modelo predeterminado para evaluar su rendimiento sobre datos reales (precisión, matriz de confusión, F1-score...).

- 🧩 **Clustering No Supervisado (DBSCAN)**  
  Explora la estructura de los datos sin etiquetas usando t-SNE + DBSCAN.

- 🧩 **Detección de Anomalías (Isolation Forest)**  
  Identifica observaciones atípicas en el conjunto de datos.

- 🔁 **Simulación de tráfico con Kafka**  
  Se utiliza `KafkaProducer` para enviar datos en lotes, simulando tráfico en tiempo real.

---

## 📁 Estructura del proyecto

```
tfg_app/
├── app/
│   ├── data/                     # Datos como X_test_filtrado.csv
│   ├── logic/
│   │   ├── load_data.py          # Carga de datos para cada módulo
│   │   ├── producer.py           # KafkaProducer en background
│   │   ├── consumer.py           # KafkaConsumer para recibir datos
│   ├── models/                   # Modelos .pkl (no se suben al repo)
│   ├── views/
│   │   ├── inicio.py             # Vista principal con Streamlit
│   │   ├── show_architecture_2.py
│   │   ├── show_dbscan.py
│   │   ├── show_isolation.py
├── docker-compose.yml           # Kafka + Zookeeper
├── requirements.txt             # Librerías necesarias
├── .gitignore                   # Archivos y carpetas ignoradas
├── app.py                       # Punto de entrada de la app
```

---

## 💻 Requisitos

- Python 3.10 o 3.11
- Docker y Docker Compose (para Kafka)
- Navegador moderno

---

## ⚙️ Instalación

### 1. Clona el repositorio

```bash
git clone https://github.com/rafaelSanchezNavarro/tfg_app.git
cd tfg_app
```

### 2. Crea y activa un entorno virtual

```bash
python -m venv venv
# En Windows CMD
venv\Scripts\activate
# En Git Bash / WSL / Linux / macOS
source venv/bin/activate
```

### 3. Instala las dependencias

```bash
pip install -r requirements.txt
```

---

## 🚀 Iniciar la aplicación

### A. Levantar Kafka y Zookeeper con Docker

```bash
docker-compose up -d
```

Esto iniciará:
- Zookeeper en `localhost:2181`
- Kafka en `localhost:9092`

Verifica con:

```bash
docker ps
```

### B. Ejecutar la app con Streamlit

```bash
streamlit run app.py
```

Accede en tu navegador a: [http://localhost:8501](http://localhost:8501)

---

## 🧪 Simulación de tráfico con Kafka

Cuando se entra a la sección de **Clasificación Supervisada**, se inicia automáticamente un `KafkaProducer` que:

- Lee `X_test_filtrado.csv`
- Envía lotes de datos simulando tráfico en tiempo real

> No es necesario abrir otra terminal: el productor corre en background con `threading`.

---

## 🧠 Modelos predeterminados

La app incluye funciones para usar modelos `.pkl` predeterminados ubicados en `app/models/`, como:

- `DecisionTreeClassifier_0.9940.pkl`
- `dbscan_model.pkl`
- `iso_forest_todo.pkl`

Puedes elegir usar estos modelos sin necesidad de cargarlos manualmente.

---

## 🔐 Buenas prácticas

- Los archivos `.pkl` y `/app/models/` están listados en `.gitignore` para evitar subir archivos grandes.
- Puedes usar Git LFS si necesitas versionar modelos grandes (opcional).
- Usa `st.session_state` para controlar interacciones y evitar cargas múltiples.

---

## 🧼 Problemas comunes y soluciones

| Error                                 | Causa                                                  | Solución                                                   |
|--------------------------------------|---------------------------------------------------------|------------------------------------------------------------|
| `NoBrokersAvailable`                 | Kafka no está corriendo en `localhost:9092`            | Asegúrate de que `docker-compose up -d` está ejecutado     |
| `File > 100MB` al hacer `git push`   | GitHub no permite archivos tan grandes                 | Usa `.gitignore` o Git LFS                                 |
| Streamlit muestra `ModuleNotFound`  | Falta una librería en el entorno virtual               | Ejecuta `pip install -r requirements.txt`                  |

---

## ✍️ Autor

Rafael Sánchez Navarro  
Repositorio: [github.com/rafaelSanchezNavarro/tfg_app](https://github.com/rafaelSanchezNavarro/tfg_app)

---

## 📜 Licencia

Este proyecto se presenta como parte de un Trabajo de Fin de Grado y su uso está permitido con fines educativos.
