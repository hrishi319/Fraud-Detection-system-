from confluent_kafka import Producer
import pandas as pd
import json
import time
import uuid

df = pd.read_excel("data/transactions_real.xlsx")

# Frequency encode a column
def freq_encode(series, value):
    counts = series.value_counts(normalize=True)
    return float(counts.get(value, 0.0))

producer = Producer({
    'bootstrap.servers': '127.0.0.1:9092'
})

def delivery_report(err, msg):
    if err:
        print(f"Delivery failed: {err}")
    else:
        print(f"Sent to {msg.topic()} [{msg.partition()}]")

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
            "transactions",
            value=json.dumps(data, default=str).encode('utf-8'),
            callback=delivery_report
        )

        producer.poll(0)
        print(f"Sent: {data['transaction_id']}")
        time.sleep(1)

    producer.flush()