import os
import time
import json
import pandas as pd
from kafka import KafkaProducer

# Cargar los datos
path_data = os.path.join('app', 'data')
X_test = pd.read_csv(os.path.join(path_data, "X_test_filtrado.csv"))
# Inicializar KafkaProducer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Simulación de tráfico
def simulate_traffic(data, batch_size=16):
    for i in range(0, len(data), batch_size):
        print(f"Enviando registros {i} a {i+batch_size}")
        batch = data.iloc[i:i+batch_size].to_dict(orient='records')
        for idx, row in enumerate(batch, start=i):
            producer.send('trafico', {'data': row, 'index': idx})  # Envía los datos y el índice al tópico 'trafico'
        time.sleep(3)  # Controla la velocidad de simulación

if __name__ == "__main__":
    while True:
        simulate_traffic(X_test, 16)  # Simula tráfico en lotes de 16 registros

