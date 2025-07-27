# Bottom Predictor API

This is a simple Flask-based API for predicting and simulating trends, profits, and prices, likely for cryptocurrency (e.g., Bitcoin) analysis. The API provides a dummy endpoint for demonstration or prototyping purposes.

## Features
- Single endpoint `/predict` that returns simulated prediction and trend data.
- CORS enabled for all routes and origins.
- Easy to run locally for prototyping or frontend integration.

## Requirements
- Python 3.9+
- Flask
- flask-cors

## Installation
1. (Recommended) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install Flask flask-cors
   ```

## Usage
Run the API server with:
```bash
python app.py
```
The server will start in debug mode on `http://127.0.0.1:5000/`.

## API Endpoint
### `GET /predict`
**Query Parameters:**
- `datetime` (string): Any datetime string (used for demonstration; not validated).

**Response:**
Returns a JSON object with the following fields:
- `datetime`: Echoes the input datetime string.
- `status`: Always `false` (dummy value).
- `real`: List of 25 simulated real values.
- `predicted`: List of 12 simulated predicted values.
- `best_index`: Dummy index (integer).
- `gained_profit`: Dummy float value.
- `actual_profit`: Dummy float value.
- `bitcoin_trend`: List of 25 simulated trend values.
- `buy_bitcoin_trend`: List of 25 simulated buy trend values.
- `bitcoin_price_trend`: List of 25 simulated price trend values.

**Example Request:**
```
GET http://127.0.0.1:5000/predict?datetime=2024-06-01T12:00:00
```

**Example Response:**
```
{
  "datetime": "2024-06-01T12:00:00",
  "status": false,
  "real": [...],
  "predicted": [...],
  "best_index": 18,
  "gained_profit": 67.9,
  "actual_profit": 89.1,
  "bitcoin_trend": [...],
  "buy_bitcoin_trend": [...],
  "bitcoin_price_trend": [...]
}
```

## Notes
- This API is for prototyping and returns random/dummy data.
- For production, replace the dummy logic with real prediction algorithms and data sources. 