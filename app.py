from flask import Flask, render_template, jsonify
import random
from datetime import datetime

app = Flask(__name__)

users = [
    "Rahul", "Priya", "Amit", "Sneha", "Siddhesh",
    "Rohan", "Neha", "Anjali", "Karan", "Pooja"
]

location_country_map = {
    "Mumbai": "India",
    "Pune": "India",
    "Delhi": "India",
    "Bangalore": "India",
    "Chennai": "India",
    "Hyderabad": "India",
    "Kolkata": "India",
    "New York": "USA",
    "London": "UK",
    "Berlin": "Germany",
    "Dubai": "UAE",
    "Singapore": "Singapore"
}

locations = list(location_country_map.keys())
devices = ["Mobile", "Laptop", "Tablet", "Desktop"]

stats = {
    "total": 0,
    "flagged": 0,
    "safe": 0,
    "review": 0
}


def generate_transaction():
    location = random.choice(locations)

    return {
        "id": "TXN" + str(random.randint(100000, 999999)),
        "user": random.choice(users),
        "amount": random.randint(500, 50000),
        "location": location,
        "country": location_country_map[location],
        "device": random.choice(devices),
        "time": datetime.now().strftime("%I:%M:%S %p")
    }


def fraud_strategy(txn):
    if txn["amount"] > 35000:
        return {
            "risk": "High",
            "status": "Fraud",
            "color": "red",
            "rowClass": "fraud-row"
        }

    if txn["country"] != "India" and txn["amount"] > 20000:
        return {
            "risk": "High",
            "status": "Fraud",
            "color": "red",
            "rowClass": "fraud-row"
        }

    if txn["device"] == "Desktop" and txn["amount"] > 22000:
        return {
            "risk": "Medium",
            "status": "Review",
            "color": "orange",
            "rowClass": "review-row"
        }

    return {
        "risk": "Low",
        "status": "Safe",
        "color": "green",
        "rowClass": "safe-row"
    }


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get_transaction")
def get_transaction():
    txn = generate_transaction()
    result = fraud_strategy(txn)

    stats["total"] += 1

    if result["status"] == "Fraud":
        stats["flagged"] += 1
    elif result["status"] == "Review":
        stats["review"] += 1
    else:
        stats["safe"] += 1

    return jsonify({
        "transaction": txn,
        "result": result,
        "stats": stats
    })


if __name__ == "__main__":
    app.run(debug=True)