# app/__init__.py

from flask import Flask
from flask_cors import CORS
from .extensions import db
from .predictions.routes import predictions_btc
from .artifacts import load_artifacts

def create_app(config_object="app.config.ProductionConfig"):
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(config_object)

    # initialize DB, CORS, etc.
    db.init_app(app)
    print("Database Initialized")
    # register blueprints
    app.register_blueprint(predictions_btc)

    with app.app_context():
    # attach artifacts for easy access in views
        ts, gts, xgb, ds, tft = load_artifacts()
    app.trend_scaler = ts
    app.google_trend_scaler = gts
    app.model_xgb    = xgb
    app.train_dataset = ds
    app.tft_model    = tft

    return app
