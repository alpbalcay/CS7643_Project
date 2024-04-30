import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.dataprocessing import Pipeline
from darts.models.filtering.moving_average_filter import MovingAverageFilter as MovingAverage
from darts.dataprocessing.transformers import InvertibleMapper, StaticCovariatesTransformer
import sklearn.preprocessing

class SalesMovingAverage:
    def __init__(self, train_merged):
        self.train_merged = train_merged
        self.family_list = train_merged['family'].unique()
        self.family_TS_dict = {}
        self.family_pipeline_dict = {}
        self.family_TS_transformed_dict = {}
        self.sales_moving_averages_dict = {}

    def create_family_TS_dict(self):
        for family in self.family_list:
            df_family = self.train_merged.loc[self.train_merged["family"] == family]
            list_of_TS_family = TimeSeries.from_group_dataframe(
                df_family,
                time_col="date",
                group_cols=["store_nbr", "family"],
                static_cols=["city", "state", "type", "cluster"],
                value_cols="sales",
                fill_missing_dates=True,
                freq="D"
            )

            for ts in list_of_TS_family:
                ts = ts.astype(np.float32)

            list_of_TS_family = sorted(
                list_of_TS_family, key=lambda ts: int(ts.static_covariates_values()[0, 0])
            )
            self.family_TS_dict[family] = list_of_TS_family

    def transform_family_TS(self):
        for key, ts_list in self.family_TS_dict.items():
            train_pipeline = Pipeline([
                MissingValuesFiller(verbose=False, n_jobs=-1),
                StaticCovariatesTransformer(
                    verbose=False,
                    transformer_cat=sklearn.preprocessing.OneHotEncoder()
                ),
                InvertibleMapper(np.log1p, np.expm1, verbose=False, n_jobs=-1),
                Scaler(verbose=False, n_jobs=-1)
            ])
            training_transformed = train_pipeline.fit_transform(ts_list)
            self.family_pipeline_dict[key] = train_pipeline
            self.family_TS_transformed_dict[key] = training_transformed

    def compute_moving_averages(self):
        sales_moving_average_7 = MovingAverage(window=7)
        sales_moving_average_28 = MovingAverage(window=28)
        
        for key, ts_list in self.family_TS_transformed_dict.items():
            sales_mas_family = []
            for ts in ts_list:
                ma_7 = sales_moving_average_7.filter(ts)
                ma_7 = TimeSeries.from_series(ma_7.pd_series())  
                ma_7 = ma_7.astype(np.float32)
                ma_7 = ma_7.with_columns_renamed(col_names=ma_7.components, col_names_new="sales_ma_7")
                ma_28 = sales_moving_average_28.filter(ts)
                ma_28 = TimeSeries.from_series(ma_28.pd_series())  
                ma_28 = ma_28.astype(np.float32)
                ma_28 = ma_28.with_columns_renamed(col_names=ma_28.components, col_names_new="sales_ma_28")
                mas = ma_7.stack(ma_28)
                sales_mas_family.append(mas)
            self.sales_moving_averages_dict[key] = sales_mas_family

    def process(self):
        self.create_family_TS_dict()
        self.transform_family_TS()
        self.compute_moving_averages()
        return {
            'family_TS_transformed_dict': self.family_TS_transformed_dict,
            'sales_moving_averages_dict': self.sales_moving_averages_dict
        }

