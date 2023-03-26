import math

import pandas as pd
import numpy as np

from .process import ProcessBase


class WeeklyIDProcess(ProcessBase):
    def __init__(self, key: str = "wid", freq: int = 30, time_column: str = "index"):
        super().__init__(key)
        self.freq = freq
        self.time_column = time_column
        self.min_factor = 0
        if self.freq < 60:
            self.min_factor = 60 // self.freq

    def run(self, data):
        org_index = data.index
        if self.time_column == "index" and isinstance(data.index, pd.DatetimeIndex):
            time_index = data.index
        else:
            time_index = pd.DatetimeIndex(data[self.time_column])
        time_df = (time_index.weekday * 24 + time_index.hour + time_index.minute / 60 * self.min_factor).to_frame().convert_dtypes(int)
        time_df.columns = [self.time_column]
        time_df.index = org_index
        return pd.concat([data, time_df], axis=1)


class DailyIDProcess(ProcessBase):
    def __init__(self, key: str = "did", freq: int = 30, time_column="index"):
        super().__init__(key)
        self.freq = freq
        self.time_column = time_column
        self.min_factor = 0
        if self.freq < 60:
            self.min_factor = 60 // self.freq

    def run(self, data: pd.DataFrame):
        org_index = data.index
        if self.time_column == "index" and isinstance(data.index, pd.DatetimeIndex):
            time_index = data.index
        else:
            time_index = pd.DatetimeIndex(data[self.time_column])
        time_df = (time_index.hour + time_index.minute / 60 * self.min_factor).to_frame().convert_dtypes(int)
        time_df.columns = [self.time_column]
        time_df.index = org_index
        return pd.concat([data, time_df], axis=1)


class SinProcess(ProcessBase):
    def __init__(self, key: str = "sin", freq: int = 60 * 24, time_column="index", amplifier=1):
        super().__init__(key)
        hours = freq // 60
        self.daily_frequency = 1 / (hours * 3600)
        self.daily_phase = 0
        self.amp = amplifier
        self.time_column = time_column

    def run(self, data):
        org_index = data.index
        if self.time_column == "index" and isinstance(data.index, pd.DatetimeIndex):
            time_df = data.index.astype("int64") // 10**9
            time_df = time_df.to_frame()
        else:
            time_df = data[self.time_column].astype("int64") // 10**9
        # time_df = self.amp * time_df.apply(lambda x: math.sin(2 * math.pi * x * self.daily_frequency + self.daily_phase))
        time_df = self.amp = np.sin(2 * math.pi * time_df * self.daily_frequency + self.daily_phase)
        time_df.index = org_index
        time_df.columns = [self.time_column]
        return pd.concat([data, time_df], axis=1)
