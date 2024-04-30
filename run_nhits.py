import pickle
import numpy as np
import pandas as pd
from models import NHitsModel
from darts.metrics import rmsle

INPUT_CHUNK_LENGTH = 266
OUTPUT_CHUNK_LENGTH = 16
VAL_LEN = 16 # days
NUM_STACKS = 3
NUM_BLOCK = 3
NUM_LAYERS = 2
LAYER_EXP = 8
BATCH_SIZE = 128
N_EPOCH = 1
DROPOUT = 0.01
MAX_SAMPLES_PER_TS = 180
LR = 0.002996870143374216


# Load Model Covariates and Inputs
with open('./data/NHiTS_covariates.pkl', 'rb') as f:
    NHiTS_covariates = pickle.load(f)
with open('./data/NHiTS_training_transformed.pkl', 'rb') as f:
    training_transformed = pickle.load(f)
with open('./data/NHiTS_train.pkl', 'rb') as f:
    NHiTS_train = pickle.load(f)


val_set = [s[-((2 * VAL_LEN) + INPUT_CHUNK_LENGTH) : -VAL_LEN] for s in training_transformed]


nhits_model = NHitsModel(batch_size=BATCH_SIZE, lr=LR, 
                         n_epoch=N_EPOCH, num_stack=NUM_STACKS, 
                         num_blocks=NUM_BLOCK, num_layers=NUM_LAYERS, 
                         layer_exp=LAYER_EXP, dropout=DROPOUT)


model = nhits_model.build(input_chunk_length=INPUT_CHUNK_LENGTH,
                          output_chunk_length=OUTPUT_CHUNK_LENGTH)


model.fit(
        series=NHiTS_train,
        val_series=val_set,
        past_covariates=NHiTS_covariates,
        val_past_covariates=NHiTS_covariates,
        max_samples_per_ts=MAX_SAMPLES_PER_TS,
        num_loader_workers=4,
    )

# Reload best model
model = model.load_from_checkpoint("nhits_model")

# Generate Forecasts for the Test Data
test_data = [ts[:-16] for ts in training_transformed]

preds = model.predict(series=test_data, past_covariates=NHiTS_covariates, n=VAL_LEN)

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

NHiTS_rmsle = rmsle(actual_series = actual_series,
                   pred_series = forecasts_back,
                   n_jobs = -1,
                   series_reduction=np.mean)

print("The mean RMSLE for the Global NHiTS Model over 1782 series is {:.5f}.".format(NHiTS_rmsle))
