from confluent_kafka import Consumer
import json
import requests
import time
import os

KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
API_URL      = os.getenv("API_URL", "http://127.0.0.1:8000")

def create_consumer():
    retries = 0
    while retries < 10:
        try:
            consumer = Consumer({
                'bootstrap.servers': KAFKA_BROKER,
                'group.id': 'fraud-detection-group',
                'auto.offset.reset': 'earliest'
            })
            consumer.subscribe(['transactions'])
            print(f"Kafka Consumer Connected to {KAFKA_BROKER}!")
            return consumer
        except Exception as e:
            print(f"Waiting for Kafka... attempt {retries+1}/10 ({e})")
            retries += 1
            time.sleep(5)
    raise Exception("Could not connect to Kafka after 10 attempts")

consumer = create_consumer()
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
            f"{API_URL}/predict",
            json=data
        )
        result = response.json()
        print("=" * 60)
        print("TRANSACTION:", result["transaction_id"])
        print("FRAUD PROBABILITY:", result["fraud_probability"])
        print("RISK LEVEL:", result["risk_level"])
        print("PREDICTION:", result["prediction"])
        print("=" * 60)

    except Exception as e:
        print("Error:", e)
        