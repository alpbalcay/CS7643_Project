import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.models.filtering.moving_average_filter import MovingAverageFilter as MovingAverage
from datetime import timedelta


class TransactionsCovariate:
    def __init__(self, df_transactions):
        self.df_transactions = df_transactions

    def preprocess(self):
        self.df_transactions.sort_values(["store_nbr", "date"], inplace=True)
        return TimeSeries.from_group_dataframe(
            self.df_transactions,
            time_col="date",
            group_cols=["store_nbr"],
            value_cols="transactions",
            fill_missing_dates=True,
            freq='D')

    def remove_static_covariates(self, ts_transactions_list):
        transactions_list = []
        for ts in ts_transactions_list:
            series = TimeSeries.from_series(ts.pd_series())
            series = series.astype(np.float32)
            transactions_list.append(series)
        return transactions_list

    def add_missing_dates(self, transactions_list):
        transactions_list_full = []
        for ts in transactions_list:
            if ts.start_time() > pd.Timestamp('20130101'):
                end_time = (ts.start_time() - timedelta(days=1))
                delta = end_time - pd.Timestamp('20130101')
                zero_series = TimeSeries.from_times_and_values(
                    times=pd.date_range(start=pd.Timestamp('20130101'),
                                        end=end_time, freq="D"),
                    values=np.zeros(delta.days + 1))
                ts = zero_series.append(ts)
                transactions_list_full.append(ts)
        return transactions_list_full

    def apply_transformations(self, transactions_list_full):
        transactions_pipeline = Pipeline([
            MissingValuesFiller(verbose=False, n_jobs=-1),
            Scaler(verbose=False, n_jobs=-1)
        ])
        return transactions_pipeline.fit_transform(transactions_list_full)

    def compute_moving_averages(self, transactions_transformed):
        trans_moving_average_7 = MovingAverage(window=7)
        trans_moving_average_28 = MovingAverage(window=28)
        transactions_covs = []

        for ts in transactions_transformed:
            ma_7 = trans_moving_average_7.filter(ts).astype(np.float32)
            ma_7 = ma_7.with_columns_renamed(col_names=ma_7.components, col_names_new="transactions_ma_7")
            ma_28 = trans_moving_average_28.filter(ts).astype(np.float32)
            ma_28 = ma_28.with_columns_renamed(col_names=ma_28.components, col_names_new="transactions_ma_28")
            trans_and_mas = ts.with_columns_renamed(col_names=ts.components, col_names_new="transactions").stack(ma_7).stack(ma_28)
            transactions_covs.append(trans_and_mas)
        return transactions_covs

    def process(self):
        ts_transactions_list = self.preprocess()
        transactions_list = self.remove_static_covariates(ts_transactions_list)
        # Handle store 24 separately!
        transactions_list[24] = transactions_list[24].slice(start_ts=pd.Timestamp('20130102'), end_ts=pd.Timestamp('20170815'))
        transactions_list_full = self.add_missing_dates(transactions_list)
        transactions_transformed = self.apply_transformations(transactions_list_full)
        transactions_covs = self.compute_moving_averages(transactions_transformed)
        return transactions_covs

