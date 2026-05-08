from confluent_kafka import Consumer
import json
import requests

consumer = Consumer({
    'bootstrap.servers': '127.0.0.1:9092',
    'group.id': 'fraud-detection-group',
    'auto.offset.reset': 'earliest'
})

consumer.subscribe(['transactions'])

print("Kafka Consumer Started...")

while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        print("Error:", msg.error())
        continue

    data = json.loads(msg.value().decode('utf-8'))

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )
        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)  # Add this line
        result = response.json()
        print("=" * 60)
        print("TRANSACTION:", result["transaction_id"])
        print("FRAUD PROBABILITY:", result["fraud_probability"])
        print("RISK LEVEL:", result["risk_level"])
        print("PREDICTION:", result["prediction"])
        print("=" * 60)

    except Exception as e:
        print("Error:", e)