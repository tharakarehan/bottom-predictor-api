# Bottom Predictor API

A Flask-based REST API for predicting and simulating cryptocurrency (e.g., Bitcoin) trends, profits, and prices using machine learning models. Designed for research, prototyping, and integration with frontend dashboards.

---

## Features
- **Prediction Endpoint**: `/btc/predict` returns model-based trend and profit predictions.
- **ML Artifacts**: Uses XGBoost and PyTorch Temporal Fusion Transformer models.
- **CORS Enabled**: For easy frontend integration.
- **Configurable**: Environment-based configuration and artifact paths.
- **Extensible**: Modular codebase for adding new endpoints or models.

---

## Directory Structure
```
.
├── app/                  # Main application package
│   ├── __init__.py       # App factory and setup
│   ├── config.py         # Configuration classes
│   ├── extensions.py     # Flask extensions (DB, etc.)
│   ├── artifacts.py      # Model/scaler loading utilities
│   └── predictions/      # Prediction service and routes
│       ├── __init__.py
│       ├── routes.py     # API endpoints
│       └── service.py    # Prediction logic
├── artifacts/            # ML models and scalers (see below)
├── requirements.txt      # Python dependencies
├── run.py                # App entry point
└── README.md             # This file
```

---

## Installation
### 1. Clone the Repository
```bash
git clone <repo-url>
cd bottom-predictor-api
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Artifacts
Place the following files in the `artifacts/` directory:
- `BPC_BTC.model` (XGBoost model)
- `trend_scaler.pkl` (scaler)
- `google_trend_scaler.pkl` (scaler)
- `train_dataset.pkl` (PyTorch dataset)
- `best_tft.ckpt` (TFT model checkpoint)

> **Note:** These files are required for the API to function. Contact the maintainer if you need access.

---

## Configuration
Configuration is managed via `app/config.py` and environment variables.

- `DATABASE_URI` (default: `mysql://root:@localhost:3306/btc_bottom`)
- `ARTIFACTS_DIR` (default: `artifacts/`)
- Model/scaler paths are set relative to `ARTIFACTS_DIR`.
- See `app/config.py` for advanced options (look-back window, prediction window, etc).

---

## Usage
### Development Server
```bash
python run.py
```
- Runs on `http://127.0.0.1:5001/` by default.

### Production (Recommended)
Use a WSGI server like Gunicorn:
```bash
gunicorn run:app
```

---

## API Reference
### `GET /btc/predict`
**Query Parameters:**
- `datetime` (string, required): Datetime string (e.g., `2024-06-01T12:00:00`).

**Response:**
Returns a JSON object with fields such as:
- `datetime`: Echoes input
- `status`: Model prediction (0/1)
- `real`: List of real values
- `predicted`: List of predicted values
- `best_index`: Integer index
- `gained_profit`: Float
- `actual_profit`: Float
- `bitcoin_trend`, `buy_bitcoin_trend`, `bitcoin_price_trend`: Lists of trend/price values

**Example Request:**
```
GET http://127.0.0.1:5001/btc/predict?datetime=2024-06-01T12:00:00
```

**Example Response:**
```json
{
  "datetime": "2024-06-01T12:00:00",
  "status": 1,
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

---

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## License
[MIT](LICENSE) (or specify your license here)

---

## Contact
Maintainer: [Your Name] (<your.email@example.com>)

---

## Notes
- This API is for research and prototyping. Predictions are only as good as the models and data provided.
- For production, ensure you secure the API, use robust error handling, and monitor performance. 