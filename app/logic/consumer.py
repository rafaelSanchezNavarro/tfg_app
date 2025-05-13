import json
from kafka import KafkaConsumer
import time

from kafka import KafkaConsumer
import json

# Función para consumir datos desde Kafka
def consume_traffic_from_kafka(batch_size=16):
    consumer = KafkaConsumer(
        'trafico',  # El nombre del tópico en Kafka
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    batch = []  # Para almacenar los mensajes del lote
    for message in consumer:
        traffic_data = message.value['data']  # Los datos reales del mensaje
        index = message.value['index']  # El índice enviado junto con los datos
        batch.append((traffic_data, index))  # Agregar el mensaje al lote

        # Cuando alcanzamos el tamaño del lote, procesamos el batch
        if len(batch) == batch_size:
            yield batch  # Devolvemos el lote de datos
            batch = []  # Limpiar el lote para el siguiente conjunto de datos

        # Opcionalmente, puedes agregar un `time.sleep(1)` si es necesario controlar la tasa de consumo