import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.models.filtering.moving_average_filter import MovingAverageFilter as MovingAverage
from tqdm import tqdm

class PromotionCovariate:
    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test
        self.family_list = df_train['family'].unique()
        self.family_promotion_dict = {}
        self.promotion_transformed_dict = {}

    def merge_and_sort(self):
        df_promotion = pd.concat([self.df_train, self.df_test], axis=0)
        return df_promotion.sort_values(["store_nbr", "family", "date"])

    def create_family_promotion_dict(self, df_promotion):
        for family in self.family_list:
            df_family = df_promotion.loc[df_promotion["family"] == family]
            list_of_TS_promo = TimeSeries.from_group_dataframe(
                df_family,
                time_col="date",
                group_cols=["store_nbr", "family"],
                value_cols="onpromotion",
                fill_missing_dates=True,
                freq="D",
            )

            for ts in list_of_TS_promo:
                ts = ts.astype(np.float32)

            self.family_promotion_dict[family] = list_of_TS_promo

    def transform_promotion_data(self):
        for key, ts_list in self.family_promotion_dict.items():
            promo_pipeline = Pipeline([
                MissingValuesFiller(verbose=False, n_jobs=-1),
                Scaler(verbose=False, n_jobs=-1)
            ])
            promotion_transformed = promo_pipeline.fit_transform(ts_list)
            self.promotion_transformed_dict[key] = self.calculate_moving_averages(promotion_transformed)

    def calculate_moving_averages(self, promotion_transformed):
        promotion_covs = []
        ma_7_transformer = MovingAverage(window=7)
        ma_28_transformer = MovingAverage(window=28)

        for ts in promotion_transformed:
            ma_7 = ma_7_transformer.filter(ts).astype(np.float32)
            ma_28 = ma_28_transformer.filter(ts).astype(np.float32)
            ma_7 = ma_7.with_columns_renamed(ma_7.components, "promotion_ma_7")
            ma_28 = ma_28.with_columns_renamed(ma_28.components, "promotion_ma_28")
            promo_and_mas = ts.stack(ma_7).stack(ma_28)
            promotion_covs.append(promo_and_mas)
        return promotion_covs

    def process(self):
        df_promotion = self.merge_and_sort()
        self.create_family_promotion_dict(df_promotion)
        self.transform_promotion_data()
        return self.promotion_transformed_dict
