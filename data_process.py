import pandas as pd
import pickle
from preprocess import (HolidaysCovariate, PromotionCovariate, 
                        OilCovariate, TimeCovariate, TransactionsCovariate,
                        SalesMovingAverage, TrainTransform)
from utils import flatten
from tqdm import tqdm

model = "LSTM"

BASE_PATH = "./data"


# Load dataset
df_train = pd.read_csv(BASE_PATH + '/train.csv')
df_test = pd.read_csv(BASE_PATH + '/test.csv')
df_holidays_events = pd.read_csv(BASE_PATH + '/holidays_events.csv')
df_oil = pd.read_csv(BASE_PATH + '/oil.csv')
df_stores = pd.read_csv(BASE_PATH + '/stores.csv')
df_transactions = pd.read_csv(BASE_PATH + '/transactions.csv')
df_sample_submission = pd.read_csv(BASE_PATH + '/sample_submission.csv')

train_merged = pd.merge(df_train, df_stores, on ='store_nbr')
train_merged = train_merged.sort_values(["store_nbr","family","date"])
train_merged = train_merged.astype({"store_nbr":'str', "family":'str', "city":'str',
                          "state":'str', "type":'str', "cluster":'str'})

FAMILY_LIST = df_train['family'].unique()

# Calculate Covariates
comp_holiday_cov = HolidaysCovariate(df_holidays_events=df_holidays_events, df_stores=df_stores)
holiday_cov = comp_holiday_cov.process()

comp_promo_cov = PromotionCovariate(df_train=df_train, df_test=df_test)
promo_cov = comp_promo_cov.process()

comp_oil_cov = OilCovariate(df_oil=df_oil)
oil_cov = comp_oil_cov.process()

time_period = pd.date_range(start='2013-01-01', end='2017-08-31', freq='D')
comp_time_cov = TimeCovariate(full_time_period=time_period)
time_cov = comp_time_cov.process()

comp_trans_cov = TransactionsCovariate(df_transactions=df_transactions)
trans_cov = comp_trans_cov.process()

# Sales Moving Average

comp_sale_ma = SalesMovingAverage(train_merged=train_merged)
sale_ma = comp_sale_ma.process()

# Train Transform
comp_train_trans = TrainTransform(train_merged=train_merged)
train_transformed = comp_train_trans.process()

# General Covariates
general_cov = time_cov.stack(oil_cov)

# Future Covariates
future_cov = []
for store in range(54): # Loop over 54 Stores
    temp_cov = holiday_cov[store].stack(general_cov)  
    future_cov.append(temp_cov)

future_covariates_dict = {}
for key in tqdm(promo_cov):
  promotion_family = promo_cov[key]
  covariates_future = [promotion_family[i].stack(future_cov[i]) for i in range(0,len(promotion_family))]
  future_covariates_dict[key] = covariates_future

# Past Covariates
past_cov = []
temp_holiday_cov = holiday_cov.copy() # for slicing past covariates

for store in range(54): # Loop over 54 Stores
    temp_holiday_cov[store] = holiday_cov[store].slice_intersect(trans_cov[store])
    general_cov_slice = general_cov.slice_intersect(trans_cov[store])
    stacked_covariates = trans_cov[store].stack(temp_holiday_cov[store]).stack(general_cov_slice)  
    past_cov.append(stacked_covariates)

past_covariates_dict = {}
for key in tqdm(promo_cov):
    promotion_family = promo_cov[key]
    sales_mas = sale_ma[key]
    covariates_past = [promotion_family[i].slice_intersect(past_cov[i]).stack(past_cov[i].stack(sales_mas[i])) for i in range(len(promotion_family))]
    past_covariates_dict[key] = covariates_past


# Only Past Covariates
only_past_covariates_dict = {}
for key in tqdm(sale_ma):
    sales_moving_averages = sale_ma[key]
    only_past_covariates = [sales_moving_averages[i].stack(trans_cov[i]) for i in range(len(sales_moving_averages))]
    only_past_covariates_dict[key] = only_past_covariates

# Data Preparation
future_covariates_full = []
for family in FAMILY_LIST:
    future_covariates_full.append(future_covariates_dict[family])

future_covariates_full = flatten(future_covariates_full)

past_covariates_full = []
for family in FAMILY_LIST:
  past_covariates_full.append(only_past_covariates_dict[family])
    
past_covariates_full = flatten(past_covariates_full)

if model == "LSTM":

    LSTM_covariates = []
    for i in range(0,len(only_past_covariates)):
        shifted = only_past_covariates[i].shift(n=16)
        cut = future_covariates_full[i].slice_intersect(shifted)
        stacked = cut.stack(shifted)
        LSTM_covariates.append(stacked)
        
    # Slice-Intersect target and covariates after shifting
    LSTM_target = []
    for i in range(0, len(train_transformed)):
        sliced = train_transformed[i].slice_intersect(LSTM_covariates[i])
        LSTM_target.append(sliced)

    # Split in train/val/test for Tuning and Validation
    val_len = 16

    LSTM_train = [s[: -(2 * val_len)] for s in LSTM_target]
    # LSTM_val = [s[-(2 * val_len) : -val_len] for s in LSTM_target]
    # LSTM_test = [s[-val_len:] for s in LSTM_target]

    with open('./data/LSTM_covariates.pkl', 'wb') as f:
        pickle.dump(LSTM_covariates, f)
    with open('./data/LSTM_target.pkl', 'wb') as f:
        pickle.dump(LSTM_target, f)
    with open('./data/LSTM_train.pkl', 'wb') as f:
        pickle.dump(LSTM_train, f)
