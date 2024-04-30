import pickle
import numpy as np
import pandas as pd
from models import TFTModel
from darts.metrics import rmsle

INPUT_CHUNK_LENGTH = 230
OUTPUT_CHUNK_LENGTH = 16
VAL_LEN = 16 # days
LSTM_LAYERS = 3
HIDDEN_SIZE = 16
N_HEAD = 4
FULL_ATTENTION = True
HIDDEN_CONT_SIZE = 16
BATCH_SIZE = 128
N_EPOCH = 1
DROPOUT = 0.01
MAX_SAMPLES_PER_TS = 7
LR = 0.009912733600616069


# Load Model Covariates and Inputs
with open('./data/TFT_train.pkl', 'rb') as f:
    train = pickle.load(f)
with open('./data/TFT_past_cov.pkl', 'rb') as f:
    tft_past_cov = pickle.load(f)
with open('./data/TFT_fut_cov.pkl', 'rb') as f:
    tft_fut_cov = pickle.load(f)
with open('./data/training_transformed.pkl', 'rb') as f:
    training_transformed = pickle.load(f)

val_set = [s[-((2 * VAL_LEN) + INPUT_CHUNK_LENGTH) : -VAL_LEN] for s in training_transformed]

tft_model = TFTModel(batch_size=BATCH_SIZE, lr=LR, 
                     n_epoch=N_EPOCH, hidden_size=HIDDEN_SIZE, 
                     lstm_layers=LSTM_LAYERS, n_head=N_HEAD, 
                     full_attention=FULL_ATTENTION, dropout=DROPOUT,
                     hidden_continuous_size=HIDDEN_CONT_SIZE)


model = tft_model.build(input_chunk_length=INPUT_CHUNK_LENGTH,
                        output_chunk_length=OUTPUT_CHUNK_LENGTH)


model.fit(
    series=train,
    val_series=val_set,
    past_covariates=tft_past_cov,
    val_past_covariates=tft_past_cov,
    future_covariates=tft_fut_cov,
    val_future_covariates=tft_fut_cov,
    max_samples_per_ts=MAX_SAMPLES_PER_TS,
    num_loader_workers=4,
)

# Reload best model
model = model.load_from_checkpoint("tft_model")

# Generate Forecasts for the Test Data
test_data = [ts[:-16] for ts in training_transformed]

preds = model.predict(series=test_data, past_covariates=tft_past_cov, future_covariates=tft_fut_cov, n=VAL_LEN)

with open('./data/train_pipeline.pkl', 'rb') as file:
    train_pipeline = pickle.load(file)
with open('./data/actual_series.pkl', 'rb') as file:
    actual_series = pickle.load(file)

# Transform Back
forecasts_back = train_pipeline.inverse_transform(preds, partial=True)

# Zero Forecasting
for n in range(0,len(forecasts_back)):
  if (actual_series[n][:-16].univariate_values()[-14:] == 0).all():
        forecasts_back[n] = forecasts_back[n].map(lambda x: x * 0)

TFT_rmsle = rmsle(actual_series = actual_series,
                   pred_series = forecasts_back,
                   n_jobs = -1,
                   series_reduction=np.mean)

print("The mean RMSLE for the Global TFT Model over 1782 series is {:.5f}.".format(TFT_rmsle))
