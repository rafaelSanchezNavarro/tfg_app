# import threading
# import os
# import time
# import json
# import pandas as pd
# from kafka import KafkaProducer

# def start_kafka_producer():
#     path_data = os.path.join('app', 'data')
#     X_test = pd.read_csv(os.path.join(path_data, "X_test_filtrado.csv"))

#     producer = KafkaProducer(
#         bootstrap_servers='localhost:9092',
#         value_serializer=lambda v: json.dumps(v).encode('utf-8')
#     )

#     def simulate_traffic(data, batch_size=5):
#         for i in range(0, len(data), batch_size):
#             batch = data.iloc[i:i+batch_size].to_dict(orient='records')
#             for idx, row in enumerate(batch, start=i):
#                 producer.send('trafico', {'data': row, 'index': idx})
#             time.sleep(3)

#     # Lanzar en hilo separado
#     thread = threading.Thread(target=simulate_traffic, args=(X_test,))
#     thread.daemon = True  # El hilo muere cuando se cierra la app
#     thread.start()


import threading
import os
import time
import pandas as pd

# Variable global para almacenar los datos simulados por batches
simulated_traffic_batches = []

def start_simulated_traffic(batch_size=5, delay=3):
    """
    Simula el envío de tráfico dividiendo X_test en batches pequeños
    y almacenándolos en una lista compartida.
    """
    global simulated_traffic_batches

    path_data = os.path.join('app', 'data')
    X_test = pd.read_csv(os.path.join(path_data, "X_test_filtrado.csv"))

    def simulate_traffic():
        for i in range(0, len(X_test), batch_size):
            batch = X_test.iloc[i:i+batch_size]
            simulated_traffic_batches.append((batch.reset_index(drop=True), list(range(i, i + len(batch)))))
            time.sleep(delay)

    # Hilo para simular tráfico
    thread = threading.Thread(target=simulate_traffic)
    thread.daemon = True
    thread.start()
