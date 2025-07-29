# app/artifacts.py
from pytorch_forecasting import TemporalFusionTransformer
import torchmetrics
import xgboost as xgb
import joblib, torch
from flask import current_app

def load_artifacts():
    # scikit‑learn / XGB
    trend_scaler = joblib.load(current_app.config["TREND_SCALER_PATH"])
    google_trend_scaler = joblib.load(current_app.config["GOOGLE_TREND_SCALER_PATH"])
    model_xgboost = xgb.XGBClassifier()
    model_xgboost.load_model(current_app.config["MODEL_XGB_PATH"])
    # PyTorch‑Forecasting
    train_ds     = torch.load(current_app.config["TS_DATASET_PATH"], weights_only=False)
    tft_model    = load_tft_strict_cpu(current_app.config["TFT_MODEL_PATH"])
    print("Artifacts Loaded")
    return trend_scaler,google_trend_scaler, model_xgboost, train_ds, tft_model

def load_tft_strict_cpu(checkpoint_path: str) -> TemporalFusionTransformer:
    """
    Load a TFT Lightning checkpoint strictly on CPU.
    Avoids any .cpu() calls that trigger torchmetrics CUDA init.
    """
    # 1) Load full checkpoint (allow unpickle) onto CPU
    ckpt = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False
    )
    # 2) Extract Lightning hyperparameters
    hparams = ckpt.get("hyper_parameters", ckpt.get("hparams", {}))
    # 3) Instantiate fresh TFT on CPU (no GPUs involved)
    model = TemporalFusionTransformer(**hparams)
    # 4) Load weights & buffers (already on CPU)
    model.load_state_dict(ckpt["state_dict"])
    # 5) Patch every torchmetrics.Metric to live on CPU
    for module in model.modules():
        if isinstance(module, torchmetrics.Metric):
            module._device = torch.device("cpu")
    # 6) Set to inference mode (no further .cpu needed)
    model.eval()
    return model