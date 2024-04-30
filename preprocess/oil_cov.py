import numpy as np
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.models.filtering.moving_average_filter import MovingAverageFilter as MovingAverage


class OilCovariate:
    def __init__(self, df_oil):
        self.df_oil = df_oil

    def process_oil_data(self):
        # Convert oil price data to TimeSeries and transform
        oil = TimeSeries.from_dataframe(self.df_oil, 
                                        time_col='date', 
                                        value_cols=['dcoilwtico'],
                                        freq='D')
        oil = oil.astype(np.float32)

        # Create and apply the pipeline for missing value filling and scaling
        oil_filler = MissingValuesFiller(verbose=False, n_jobs=-1)
        oil_scaler = Scaler(verbose=False, n_jobs=-1)
        oil_pipeline = Pipeline([oil_filler, oil_scaler])
        oil_transformed = oil_pipeline.fit_transform(oil)
        return oil_transformed

    def compute_moving_averages(self, oil_transformed):
        # Compute moving averages for the oil price
        oil_moving_average_7 = MovingAverage(window=7)
        oil_moving_average_28 = MovingAverage(window=28)

        ma_7 = oil_moving_average_7.filter(oil_transformed).astype(np.float32)
        ma_7 = ma_7.with_columns_renamed(col_names=ma_7.components, col_names_new="oil_ma_7")
        ma_28 = oil_moving_average_28.filter(oil_transformed).astype(np.float32)
        ma_28 = ma_28.with_columns_renamed(col_names=ma_28.components, col_names_new="oil_ma_28")
        oil_moving_averages = ma_7.stack(ma_28)
        return oil_moving_averages

    def process(self):
        oil_transformed = self.process_oil_data()
        oil_moving_averages = self.compute_moving_averages(oil_transformed)
        oil_covariates = oil_transformed.stack(oil_moving_averages)
        return oil_covariates

