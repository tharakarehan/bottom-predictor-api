from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import random
real_base = 23001

app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app)

@app.route('/predict', methods=['GET'])
def predict():
    dt_str = request.args.get('datetime')
    # Dummy payload
    dummy = {
        "datetime": dt_str,
        "status": False,
        "real" : [real_base + i + random.randint(-5000, 5000) for i in range(25)],
        "predicted" :[real_base + i + random.randint(-3000, 3000) for i in range(12)],
        "best_index": 18,
        "gained_profit": 67.9,
        "actual_profit": 89.1,
        "bitcoin_trend": [50.3  * random.randint(-50, 50) for i in range(25)],
        "buy_bitcoin_trend": [10.3  * random.randint(-5, 25) for i in range(25)],
        "bitcoin_price_trend": [80.3 * random.randint(-100, 50) for i in range(25)],
    }
    return jsonify(dummy)

if __name__ == '__main__':
    # Restart server after changes
    app.run(debug=True)