# app/predictions/service.py

from flask import current_app, jsonify
from sqlalchemy import text
import torch
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import numdifftools as nd
from datetime import datetime, timedelta

class PredictionService:
    def __init__(self, db_session):
        # hold references to loaded artifacts
        self.db      = db_session
        self.trend_scaler  = current_app.trend_scaler
        self.google_trend_scaler = current_app.google_trend_scaler
        self.model_xgb = current_app.model_xgb
        self.train_ds  = current_app.train_dataset
        self.tft_model = current_app.tft_model
        self.smoothing_threshold = 50
        self.sliding_window_size = current_app.config["LOOK_BACK"]
        self.prediction_window = current_app.config["PREDICTION_WINDOW"]
        self.classification_label_size = current_app.config["CLS_LABEL_SIZE"]
        self.trend_measure_lenth = current_app.config["TREND_MEASURE_LENGTH"]
        self.include_google_trend = current_app.config["INCLUDE_GOOOGLE_TREND"]
        self.test_offset = 20000
        self.google_trend_length = current_app.config["GOOGLE_TREND_LENGTH"]

    def fetch_data_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Returns all columns from your_table where timestamp BETWEEN start_date AND end_date.
        """
        sql = text("""
            SELECT *
            FROM bitcoin_pca
            WHERE `timestamp` BETWEEN :start_date AND :end_date
        """)
        result = self.db.session.execute(
            sql,
            {"start_date": start_date, "end_date": end_date}
        )
        # .mappings().all() gives a list of RowMapping dicts
        df = pd.DataFrame(result.mappings().all())  # :contentReference[oaicite:0]{index=0}
        return df

    def preprocess(self, time_stamp, datetime_5m):
        start_timestamp_max, start_timestamp, start_timestamp_tft, end_timestamp = self.get_boundaries(datetime_5m)

        df_filtered_max = self.fetch_data_range(start_timestamp_max, end_timestamp)
        df_filtered = self.get_data_range(df_filtered_max, start_timestamp, end_timestamp)
        df_filtered_tft = self.get_data_range(df_filtered, start_timestamp_tft, end_timestamp)

        labels_df = df_filtered["close"]
        features_df = df_filtered.drop(columns=["close", "Bitcoin", "Bitcoin price", "Bitcoin dip", "Buy Bitcoin", "timestamp"])
        google_trend_df = df_filtered[["Bitcoin price", "Buy Bitcoin"]]

        test_x = features_df.iloc[:].values
        test_y = labels_df.iloc[:]
        test_gt = google_trend_df.iloc[:].values
        test_gt = self.google_trend_scaler.transform(test_gt)

        raw_test_y = test_y.values.reshape(-1, 1)
        test_direction_slide, test_google_trend = self.sliding_window(test_x, raw_test_y, raw_test_y, test_gt)

        test_trend_features = self.trend_detection(test_direction_slide)
        test_trend_features = torch.tensor(self.trend_scaler.transform(test_trend_features))
        new_feature_tensor_test = test_trend_features.repeat(1,10)
        new_feature_tensor_test = new_feature_tensor_test.unsqueeze(2)
        return df_filtered_max, df_filtered_tft, new_feature_tensor_test, test_google_trend

    def predict(self, time_stamp):
        print(time_stamp)
        datetime_5m = self.round_up_to_next_5min(time_stamp)
        df_filtered_max, df_filtered_tft, new_feature_tensor_test, test_google_trend = self.preprocess(time_stamp, datetime_5m)
        test_input_features, pred_test = self.get_intermediate_weights(df_filtered_tft)
        # Concatenate additional feature tensors
        test_input_features = torch.cat(
            (test_input_features, new_feature_tensor_test), dim=2
        )
        if self.include_google_trend:
            test_input_features = torch.cat(
                (test_input_features, test_google_trend), dim=2
            )
        # Run model
        pred_direction_test = self.xgboost_test(
            test_input_features
        )
        pred_direction_test = torch.round(pred_direction_test).long()
        response = self.get_response(df_filtered_max, pred_direction_test, pred_test, datetime_5m)
        return response
    
    @staticmethod
    def get_response(df_filtered_max, pred_direction_test, pred_test, datetime_5m):
        status = pred_direction_test[0,0].item()
        predicted = pred_test[:, 0, 1].tolist()
        real = df_filtered_max['close'].tolist()
        bitcoin_trend = df_filtered_max['Bitcoin'].tolist()
        buy_bitcoin_trend = df_filtered_max['Buy Bitcoin'].tolist()
        bitcoin_price_trend = df_filtered_max['Bitcoin price'].tolist()

        sub_list = real[12 : 25]
        min_value = min(sub_list)
        min_index_in_sublist = sub_list.index(min_value)
        min_index_in_original_list = 12 + min_index_in_sublist

        entry_price = round(real[12], 4)
        best_price = round(real[min_index_in_original_list], 4)
        exit_price = round(real[24], 4)

        response = {
            "datetime": datetime_5m,
            "status": status == 1,
            "real" : real,
            "predicted" :predicted,
            "best_index": min_index_in_original_list,
            "gained_profit": exit_price - entry_price,
            "actual_profit": exit_price - best_price,
            "bitcoin_trend": bitcoin_trend,
            "buy_bitcoin_trend": buy_bitcoin_trend,
            "bitcoin_price_trend": bitcoin_price_trend,
        }
        return jsonify(response)

    def get_boundaries(self, datetime_5m):
        look_back_max = max(self.sliding_window_size, self.trend_measure_lenth, self.classification_label_size)
        look_back = max(self.sliding_window_size, self.trend_measure_lenth)
        look_back_tft = self.sliding_window_size
        end_timestamp = self.shift_by_5_min_intervals(datetime_5m, self.classification_label_size)
        start_timestamp_max = self.shift_by_5_min_intervals(datetime_5m, -look_back_max)
        start_timestamp = self.shift_by_5_min_intervals(datetime_5m, -look_back)
        start_timestamp_tft = self.shift_by_5_min_intervals(datetime_5m, -look_back_tft)
        return start_timestamp_max, start_timestamp, start_timestamp_tft, end_timestamp
    
    @staticmethod
    def shift_by_5_min_intervals(timestamp_str: str, n: int) -> str:
        # Parse the datetime from custom format
        dt = datetime.strptime(timestamp_str.replace('+', ' '), "%Y-%m-%d %H:%M:%S")
        # Shift by n * 5 minutes
        shifted = dt + timedelta(minutes=5 * n)
        # Return in the same format with '+'
        return shifted.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def round_up_to_next_5min(timestamp_str: str) -> str:
        # Parse the input (replace '+' as space or 'T')
        # Format is "YYYY‑MM‑DD+HH:MM:SS"
        dt = datetime.strptime(timestamp_str.replace('+', ' '), "%Y-%m-%d %H:%M:%S")
        # Zero out seconds and microseconds
        dt = dt.replace(second=0, microsecond=0)
        # Compute minutes to next multiple of 5
        minutes = dt.minute
        remainder = minutes % 5
        if remainder == 0:
            # Already aligned exactly
            rounded = dt
        else:
            increment = 5 - remainder
            rounded = dt + timedelta(minutes=increment)
        # Format back with '+'
        return rounded.strftime("%Y-%m-%d+%H:%M:%S")

    @staticmethod
    def get_data_range(df, start_date, end_date):
        return df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

    def sliding_window(self, x, y, y_raw, gt):
        y_tft_tr = []
        y_tft_gt = []
        backward_length = max(self.sliding_window_size, self.trend_measure_lenth)
        for i in range(backward_length, x.shape[0]):
            tmp_y_tft_tr = y_raw[i - self.trend_measure_lenth: i, :]
            tmp_y_tft_gt = gt[i - self.google_trend_length: i, :]
            y_tft_tr.append(tmp_y_tft_tr)
            y_tft_gt.append(tmp_y_tft_gt)
        y_tft_tr = torch.from_numpy(np.array(y_tft_tr)).float()
        y_tft_gt = torch.from_numpy(np.array(y_tft_gt)).float()
        return y_tft_tr, y_tft_gt
    
    @staticmethod
    def trend_detection(data):
        n = data.shape[1]
        sets = data.shape[0]
        mean_derivatives = np.zeros((sets,1))

        data = data.numpy()

        for i in range(sets):
            # Select the i-th 20 set
            x = np.arange(1,n+1)
            x_fake = np.arange(1.1, n, 0.1)
            y = data[i, :, 0]
            # Simple interpolation of x and y
            f = interp1d(x, y)

            # derivative of y with respect to x
            df_dx = nd.Derivative(f, step=1e-6)(x_fake)
            # Calculate the mean derivative for the i-th 20 set
            average = np.average(df_dx)
            mean_derivatives[i][0] = average
        return  torch.from_numpy(mean_derivatives)
    
    def infer_attention_batch(
        self,
        df: pd.DataFrame,
    ):
        print("train_dataset loaded")
        dataset = TimeSeriesDataSet.from_dataset(
        self.train_ds,
        df,
        predict=False,
        stop_randomization=True
        )
        dl = dataset.to_dataloader(train=False, batch_size=32, num_workers=0)
        enc_lstm_list = []
        post_lstm_enc_list = []

        def lstm_enc_hook(module, inp, out):
            enc_lstm_list.append(out[0].detach())          # [B, enc_len, H]

        def post_lstm_hook(module, inp, out):
            # Keep only encoder-length outputs
            if out.size(1) == self.sliding_window_size:
                post_lstm_enc_list.append(out.detach())    # [B, enc_len, H]

        h1 = self.tft_model.lstm_encoder.register_forward_hook(lstm_enc_hook)
        h2 = self.tft_model.post_lstm_add_norm_encoder.register_forward_hook(post_lstm_hook)

        try:
            raw = self.tft_model.predict(dl, mode="raw")
        finally:
            h1.remove(); h2.remove()

        encoder_attention = raw["encoder_attention"].cpu()
        decoder_attention = raw["decoder_attention"].cpu()
        prediction        = raw["prediction"].cpu()

        enc_lstm_seq  = torch.cat(enc_lstm_list, dim=0).cpu()          # [N, enc_len, H]
        post_lstm_seq = torch.cat(post_lstm_enc_list, dim=0).cpu()     # aligned shapes

        return encoder_attention, decoder_attention, prediction, enc_lstm_seq, post_lstm_seq

    def get_intermediate_weights(self, chunk):
        chunk["time_idx"] = chunk.index
        chunk["series"]   = "BTC"
        enc_attn, dec_attn, pred_1, enc_lstm_seq, post_lstm_seq = self.infer_attention_batch(
                chunk
            )
        return post_lstm_seq, pred_1
    
    def xgboost_test(self, x):
        tft_seq_xg = x.reshape(x.shape[0], -1)
        pred = self.model_xgb.predict(tft_seq_xg.detach().cpu().numpy())
        pred = torch.tensor(pred).reshape(pred.shape[0],1)
        return pred
