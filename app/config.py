# app/config.py

import os

class BaseConfig:
    # Pull from environment or default
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URI",
        "mysql://root:@localhost:3306/btc_bottom"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # to suppress warnings

    # absolute path to the directory containing this file
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # :contentReference[oaicite:0]{index=0}

    # directory where all serialized artifacts live;
    # can be overridden via the ARTIFACTS_DIR env var
    ARTIFACTS_DIR = os.getenv(
        "ARTIFACTS_DIR",
        os.path.join(BASE_DIR, "..", "artifacts")
    )  # :contentReference[oaicite:1]{index=1}

    # scikit‐learn / XGBoost
    MODEL_XGB_PATH       = os.path.join(ARTIFACTS_DIR, "BPC_BTC.model")
    TREND_SCALER_PATH    = os.path.join(ARTIFACTS_DIR, "trend_scaler.pkl")
    GOOGLE_TREND_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "google_trend_scaler.pkl")

    # PyTorch‐Forecasting dataset & TFT model
    TS_DATASET_PATH      = os.path.join(ARTIFACTS_DIR, "train_dataset.pkl")
    TFT_MODEL_PATH       = os.path.join(ARTIFACTS_DIR, "best_tft.ckpt")

    LOOK_BACK = 10
    PREDICTION_WINDOW = 1
    CLS_LABEL_SIZE = 12
    TREND_MEASURE_LENGTH = 12
    INCLUDE_GOOOGLE_TREND = True
    # test_offset = 20000
    GOOGLE_TREND_LENGTH = LOOK_BACK

class DevelopmentConfig(BaseConfig):
    DEBUG = True

class ProductionConfig(BaseConfig):
    DEBUG = False
