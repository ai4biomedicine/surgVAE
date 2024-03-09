import numpy as np
import pandas as pd
class InterpolatorDenseTS:
    def __init__(self, mean, std, ts):
        self.means = mean
        self.stds = std.replace(0,1)
        self.ts = ts

    def __call__(self, id, end_time):
        # first column is time index, all following columns are features
        data = self.ts.loc[
            self.ts.orlogid_encoded == id, self.ts.columns != "orlogid_encoded"
        ]
        df = pd.DataFrame(columns=data.columns)
        
        self.start_time = 0
        self.end_time = end_time 
        df.timepoint = np.arange(self.start_time, self.end_time + 1)
        df[df.timepoint.isin(data.timepoint)] = data[
        data.timepoint.isin(df.timepoint)
        ].values
        df = df.astype(float)
        df.interpolate(method="linear", inplace=True)
        df.fillna(0, inplace=True)
        df.iloc[:, 1:] = (df.iloc[:, 1:] - self.means) / self.stds
        return df

class InterpolatorScarseTS:
    def __init__(self, mean, std, ts, measures):
        self.means = mean
        self.stds = std
        self.num_measures = measures
        self.ts = ts

    def _flatten_columns(self, data):
        columns = ["timepoint"]
        for i in self.num_measures:
            columns.append("measure_" + str(i))
        df = pd.DataFrame(columns=columns)
        df.timepoint = np.arange(self.start_time, self.end_time + 1)
        for index, measure in enumerate(self.num_measures):
            df_col = (
                data[data.measure_index == measure]
                .sort_values("timepoint")
                .groupby(["timepoint"])
                .mean()
                .reset_index(drop=False)
            )
            if len(df_col) > 0:
                df_col.VALUE = (
                    df_col.VALUE - self.means[self.means.index == measure].values[0]
                ) / self.stds[self.stds.index == measure].values[0]
                # try:
                df.loc[
                    df.timepoint.isin(df_col["timepoint"].values), columns[index + 1]
                ] = df_col.VALUE.values
            # except:
        return df

    def __call__(self, id, end_time):
        # first column is time index, all following columns are features
        data = self.ts.loc[
            self.ts.orlogid_encoded == id, self.ts.columns != "orlogid_encoded"
        ]  # timepoint, measure_index, VALUE
        self.start_time = 0
        self.end_time = end_time
        df = self._flatten_columns(data)
        df = df.astype(float)
        df.interpolate(method="linear", inplace=True)
        df.fillna(0, inplace=True)
        return df
    
class InterpolatorMedDoseOnly:
    def __init__(self, mean, std, ts, med_integers):
        self.means = mean
        self.stds = std
        self.med_integers = med_integers
        self.ts = ts

    def _flatten_columns(self, data):
        columns = ["time"]
        for i in self.med_integers:
            columns.append("med_" + str(i))
        df = pd.DataFrame(columns=columns)
        df.time = np.arange(self.start_time, self.end_time + 1)
        for index, med in enumerate(self.med_integers):
            df_col = (
                data[data.med_integer == med]
                .sort_values("time")
                .groupby(["time"])
                .mean()
                .reset_index(drop=False)
            )
            if len(df_col) > 0:
                df_col.dose = (
                    df_col.dose
                    - self.means[self.means.index.astype(int) == med].values[0]
                ) / self.stds[self.stds.index.astype(int) == med].values[0]
                # try:
                df.loc[
                    df.time.isin(df_col["time"].values), columns[index + 1]
                ] = df_col.dose.values
            # except:
        return df

    def __call__(self, id, end_time):
        # first column is time index, all following columns are features
        data = self.ts.loc[
            self.ts.orlogid_encoded == id, self.ts.columns != "orlogid_encoded"
        ]  # timepoint, measure_index, VALUE
        self.start_time = 0
        self.end_time = end_time
        df = self._flatten_columns(data)
        df = df.astype(float)
        df.interpolate(method="linear", inplace=True)
        df.fillna(0, inplace=True)
        return df
                
