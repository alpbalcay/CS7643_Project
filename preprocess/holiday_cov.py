import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller


class HolidaysCovariate:
    def __init__(self, df_holidays_events, df_stores):
        self.he = df_holidays_events
        self.st = df_stores

    def redefine_categories(self):
        df = self.he.copy()
        df["type"] = np.where(df["transferred"] == True, "Transferred", df["type"])
        df["type"] = np.where(df["type"] == "Transfer", "Holiday", df["type"])
        df["type"] = np.where(df["type"] == "Additional", "Holiday", df["type"])
        df["type"] = np.where(df["type"] == "Bridge", "Holiday", df["type"])
        self.he = df

    def generate_holiday_list(self):
        listofseries = []

        for i in range(0, len(self.st)):

            temp_df = pd.DataFrame(columns=["date"])
            temp_df["date"] = self.he["date"]

            temp_df["national_holiday"] = np.where(
                ((self.he["type"] == "Holiday") & (self.he["locale"] == "National")),
                1,
                0,
            )

            temp_df["earthquake_relief"] = np.where(
                self.he["description"].str.contains("Terremoto Manabi"), 1, 0
            )

            temp_df["christmas"] = np.where(
                self.he["description"].str.contains("Navidad"), 1, 0
            )

            temp_df["football_event"] = np.where(
                self.he["description"].str.contains("futbol"), 1, 0
            )

            temp_df["national_event"] = np.where(
                (
                    (self.he["type"] == "Event")
                    & (self.he["locale"] == "National")
                    & (~self.he["description"].str.contains("Terremoto Manabi"))
                    & (~self.he["description"].str.contains("futbol"))
                ),
                1,
                0,
            )

            temp_df["work_day"] = np.where((self.he["type"] == "Work Day"), 1, 0)

            temp_df["local_holiday"] = np.where(
                (
                    (self.he["type"] == "Holiday")
                    & (
                        (self.he["locale_name"] == self.st["state"][i])
                        | (self.he["locale_name"] == self.st["city"][i])
                    )
                ),
                1,
                0,
            )

            listofseries.append(temp_df)

        return listofseries

    def remove_0_and_duplicates(self, holiday_list):
        listofseries = []
    
        for i in range(0,len(holiday_list)):
                
            df_holiday_per_store = holiday_list[i].set_index('date')

            df_holiday_per_store = df_holiday_per_store.loc[~(df_holiday_per_store==0).all(axis=1)]
            
            df_holiday_per_store = df_holiday_per_store.groupby('date').agg({'national_holiday':'max', 'earthquake_relief':'max', 
                                'christmas':'max', 'football_event':'max', 
                                'national_event':'max', 'work_day':'max', 
                                'local_holiday':'max'}).reset_index()

            listofseries.append(df_holiday_per_store)


        return listofseries
    
    def holiday_TS_list_54(self, holiday_list):

        listofseries = []
        
        for i in range(0,54):
                
                holidays_TS = TimeSeries.from_dataframe(holiday_list[i], 
                                            time_col = 'date',
                                            fill_missing_dates=True,
                                            fillna_value=0,
                                            freq='D')
                
                holidays_TS = holidays_TS.slice(pd.Timestamp('20130101'),pd.Timestamp('20170831'))
                holidays_TS = holidays_TS.astype(np.float32)
                listofseries.append(holidays_TS)

        return listofseries

    def process(self):
        self.redefine_categories()
        list_of_holidays_per_store = self.generate_holiday_list()
        list_of_holidays_per_store = self.remove_0_and_duplicates(list_of_holidays_per_store)
        list_of_holidays_store = self.holiday_TS_list_54(list_of_holidays_per_store)

        holidays_pipeline = Pipeline(
            [
                MissingValuesFiller(verbose=False, n_jobs=-1, name="Filler"),
                Scaler(verbose=False, n_jobs=-1, name="Scaler"),
            ]
        )
        holidays_transformed = holidays_pipeline.fit_transform(list_of_holidays_store)
        return holidays_transformed
