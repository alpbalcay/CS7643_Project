import pickle
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller, StaticCovariatesTransformer
from darts.dataprocessing.transformers import InvertibleMapper
import sklearn.preprocessing
import numpy as np

class TrainTransform:
    def __init__(self, train_merged):
        self.train_merged = train_merged

    def create_time_series(self):
        # Create TimeSeries objects from the DataFrame
        list_of_TS = TimeSeries.from_group_dataframe(
            self.train_merged,
            time_col="date",
            group_cols=["store_nbr", "family"],
            static_cols=["city", "state", "type", "cluster"],
            value_cols="sales",
            fill_missing_dates=True,
            freq="D"
        )
        for ts in list_of_TS:
            ts = ts.astype(np.float32)
        return list_of_TS

    def sort_time_series(self, list_of_TS):
        # Sort the list of TimeSeries based on a static covariate value
        return sorted(list_of_TS, key=lambda ts: int(ts.static_covariates_values()[0, 0]))

    def apply_transformations(self, list_of_TS):
        # Define and apply the transformation pipeline
        train_filler = MissingValuesFiller(verbose=False, n_jobs=-1)
        static_cov_transformer = StaticCovariatesTransformer(
            verbose=False, 
            transformer_cat=sklearn.preprocessing.OneHotEncoder(),
            name="Encoder"
        )
        log_transformer = InvertibleMapper(np.log1p, np.expm1, verbose=False, n_jobs=-1)
        train_scaler = Scaler(verbose=False, n_jobs=-1)

        train_pipeline = Pipeline([
            train_filler,
            static_cov_transformer,
            log_transformer,
            train_scaler
        ])

        training_transformed = train_pipeline.fit_transform(list_of_TS)

        with open('./data/train_pipeline.pkl', 'wb') as file:
            pickle.dump(train_pipeline, file)

        return training_transformed

    def process(self):
        # Process the training data
        list_of_TS = self.create_time_series()
        sorted_list_of_TS = self.sort_time_series(list_of_TS)
        transformed_list_of_TS = self.apply_transformations(sorted_list_of_TS)
        return transformed_list_of_TS

