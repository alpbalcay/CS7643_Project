import pickle
import numpy as np
import pandas as pd
from models import LSTMModel
from darts.metrics import rmsle

INPUT_CHUNK_LENGTH = 131
VAL_LEN = 16 # days
BATCH_SIZE = 128
N_EPOCH = 1
HIDDEN_DIM = 39
RNN_LAYERS = 3
DROPOUT = 0
MAX_SAMPLES_PER_TS = 60
LR = 0.0019971227090605087

# Load Model Covariates and Inputs
with open('./data/LSTM_covariates.pkl', 'rb') as f:
    LSTM_covariates = pickle.load(f)
with open('./data/LSTM_target.pkl', 'rb') as f:
    LSTM_target = pickle.load(f)
with open('./data/LSTM_train.pkl', 'rb') as f:
    LSTM_train = pickle.load(f)


val_set = [s[-((2 * VAL_LEN) + INPUT_CHUNK_LENGTH) : -VAL_LEN] for s in LSTM_target]

lstm_model = LSTMModel(val_len=VAL_LEN, batch_size=BATCH_SIZE, lr=LR, n_epoch=N_EPOCH,
                  hidden_dim=HIDDEN_DIM, n_rnn_layers=RNN_LAYERS,
                  dropout=DROPOUT)

model = lstm_model.build(input_chunk_length=INPUT_CHUNK_LENGTH)

model.fit(
        series=LSTM_train,
        val_series=val_set,
        future_covariates=LSTM_covariates,
        val_future_covariates=LSTM_covariates,
        max_samples_per_ts=MAX_SAMPLES_PER_TS,
        num_loader_workers=4,
    )

# Reload best model
model = model.load_from_checkpoint("lstm_model")

# Generate Forecasts for the Test Data
test_data = [ts[:-16] for ts in LSTM_target] 
preds = model.predict(series=test_data, future_covariates=LSTM_covariates, n=VAL_LEN)

with open('./data/train_pipeline.pkl', 'rb') as file:
    train_pipeline = pickle.load(file)
with open('./data/actual_series.pkl', 'rb') as file:
    actual_series = pickle.load(file)

# Transform Back
forecasts_back = train_pipeline.inverse_transform(preds, partial=True)

# Zero Forecasting
for n in range(0,len(forecasts_back)):
  if (LSTM_target[n][:-16].univariate_values()[-14:] == 0).all():
        forecasts_back[n] = forecasts_back[n].map(lambda x: x * 0)

LSTM_rmsle = rmsle(actual_series = actual_series,
                   pred_series = forecasts_back,
                   n_jobs = -1,
                   series_reduction=np.mean)

print("The mean RMSLE for the Global LSTM Model over 1782 series is {:.5f}.".format(LSTM_rmsle))
