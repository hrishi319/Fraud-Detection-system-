from confluent_kafka import Producer
import pandas as pd
import json
import time
import uuid
import os

# ── Config ────────────────────────────────────────────────────────────────────
KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
KAFKA_TOPIC  = os.getenv("KAFKA_TOPIC", "transactions")
DATA_PATH    = os.getenv("DATA_PATH", "data/transactions_real.xlsx")

print(f"Connecting to Kafka broker: {KAFKA_BROKER}")
print(f"Publishing to topic: {KAFKA_TOPIC}")
print(f"Reading data from: {DATA_PATH}")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH)
print(f"Loaded {len(df)} transactions from Excel")


# ── Frequency encode a column ─────────────────────────────────────────────────
def freq_encode(series, value):
    counts = series.value_counts(normalize=True)
    return float(counts.get(value, 0.0))


# ── Producer with retry ───────────────────────────────────────────────────────
def create_producer(retries=10):
    for attempt in range(retries):
        try:
            p = Producer({'bootstrap.servers': KAFKA_BROKER})
            print(f"Producer connected to {KAFKA_BROKER}")
            return p
        except Exception as e:
            print(f"Waiting for Kafka... attempt {attempt+1}/{retries} ({e})")
            time.sleep(5)
    raise Exception("Could not connect to Kafka after multiple attempts")


producer = create_producer()

# ── Delivery report ───────────────────────────────────────────────────────────
def delivery_report(err, msg):
    if err:
        print(f"Delivery failed: {err}")
    else:
        print(f"Sent to {msg.topic()} [{msg.partition()}]")
        

# ── Main loop ─────────────────────────────────────────────────────────────────
print("Producer started — sending transactions...")

while True:
    for _, row in df.iterrows():

        data = {
            "transaction_id": str(uuid.uuid4()),
            "merchant"      : freq_encode(df["merchant"], row.get("merchant")),
            "category"      : freq_encode(df["category"], row.get("category")),
            "amt"           : float(row.get("amt", 0)),
            "gender"        : 0.0 if str(row.get("gender", "M")).upper() == "M" else 1.0,
            "city"          : freq_encode(df["city"], row.get("city")),
            "state"         : freq_encode(df["state"], row.get("state")),
            "zip"           : float(row.get("zip", 0)),
            "city_pop"      : float(row.get("city_pop", 0)),
            "job"           : freq_encode(df["job"], row.get("job")),
            "hour"          : float(row.get("hour", 12)),
            "day_of_week"   : float(row.get("day_of_week", 1)),
            "month"         : float(row.get("month", 1)),
            "is_weekend"    : float(row.get("is_weekend", 0)),
            "is_night"      : float(row.get("is_night", 0)),
            "age"           : float(row.get("age", 30)),
            "geo_distance"  : float(row.get("geo_distance", 0.0))
        }

        producer.produce(
            KAFKA_TOPIC,
            value=json.dumps(data, default=str).encode('utf-8'),
            callback=delivery_report
        )

        producer.poll(0)
        print(f"Sent: {data['transaction_id']}")
        time.sleep(1)

    producer.flush()
    print("Completed one full cycle — restarting...")
