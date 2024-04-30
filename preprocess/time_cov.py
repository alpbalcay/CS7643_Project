import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler

class TimeCovariate:
    def __init__(self, full_time_period):
        self.full_time_period = full_time_period

    def create_time_based_covariates(self):
        # Create time-based covariates
        year = datetime_attribute_timeseries(time_index=self.full_time_period, attribute="year")
        month = datetime_attribute_timeseries(time_index=self.full_time_period, attribute="month")
        day = datetime_attribute_timeseries(time_index=self.full_time_period, attribute="day")
        dayofyear = datetime_attribute_timeseries(time_index=self.full_time_period, attribute="dayofyear")
        weekday = datetime_attribute_timeseries(time_index=self.full_time_period, attribute="dayofweek")
        weekofyear = datetime_attribute_timeseries(time_index=self.full_time_period, attribute="weekofyear")
        timesteps = TimeSeries.from_times_and_values(
            times=self.full_time_period,
            values=np.arange(len(self.full_time_period)),
            columns=["linear_increase"]
        )
        time_cov = year.stack(month).stack(day).stack(dayofyear).stack(weekday).stack(weekofyear).stack(timesteps)
        return time_cov.astype(np.float32)

    def transform(self, time_cov):
        # Transform the time covariates
        scaler = Scaler()
        time_cov_train, _ = time_cov.split_before(pd.Timestamp('20170816'))
        scaler.fit(time_cov_train)
        return scaler.transform(time_cov)

    def process(self):
        time_cov = self.create_time_based_covariates()
        time_cov_transformed = self.transform(time_cov)
        return time_cov_transformed

