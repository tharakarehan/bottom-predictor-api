# app/predictions/routes.py

from flask import Blueprint, request
from app.predictions.service import PredictionService
from app.extensions import db

predictions_btc = Blueprint("btc", __name__, url_prefix="/btc")

@predictions_btc.route("/predict", methods=["GET"])
def get_predictions():
    print("received")
    dt_str = request.args.get('datetime')
    svc = PredictionService(db)
    results = svc.predict(dt_str)
    return results
