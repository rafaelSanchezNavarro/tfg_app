anomalies = {
    "Reconnaissance": [
        "Generic_scanning",
        "Scanning_vulnerability",
        "fuzzing",
        "Discovering_resources"
    ],
    "Weaponization": [
        "BruteForce",
        "Dictionary",
        "insider_malcious"
    ],
    "Exploitation": [
        "Reverse_shell",
        "MitM"
    ],
    "Lateral _movement": [
        "MQTT_cloud_broker_subscription",
        "Modbus_register_reading",
        "TCP Relay"
    ],
    "C&C": [
        "C&C"
    ],
    "Exfiltration": [
        "Exfiltration"
    ],
    "Tampering": [
        "False_data_injection",
        "Fake_notification"
    ],
    "crypto-ransomware": [
        "crypto-ransomware"
    ],
    "RDOS": [
        "RDOS"
    ]
}

def encontrar_categoria(valor_buscado):
    for categoria, eventos in anomalies.items():
        if valor_buscado in eventos:
            return categoria
    return None  
