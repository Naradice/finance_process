from collections.abc import Iterable
from datetime import time
from typing import Union
import json
import os
import warnings

import numpy as np
import pandas as pd

from . import convert, standalization
from .process import ProcessBase


def get_available_processes() -> dict:
    processes = {
        "Diff": DiffPreProcess,
        "MiniMax": MinMaxPreProcess,
        "STD": STDPreProcess,
    }
    return processes


def preprocess_to_params(processes: list) -> dict:
    """convert procese list to dict for saving params as file

    Args:
        processes (list: ProcessBase): preprocess defiend in preprocess.py

    Returns:
        dict: {key: params}
    """
    params = {}
    for process in processes:
        option = process.option
        option["kinds"] = process.kinds
        params[process.key] = option
    return params


def save_preprocesses(processes: list, file_name: str = None):
    params = preprocess_to_params(processes)
    if file_name is None:
        file_name = os.path.join(os.getcwd(), "preprocess.json")
    with open(file_name, mode="w") as fp:
        json.dump(params, fp)


def load_preprocess(arg: Union[str, dict]) -> list:
    if type(arg) is str:
        with open(arg, mode="r") as fp:
            params = json.load(fp)
    elif type(arg) is dict:
        params = arg
    else:
        raise TypeError(f"argument should be str or dict. {type(arg)} is provided.")
    ips_dict = get_available_processes()
    pss = []
    for key, param in params.items():
        kinds = param.pop("kinds")
        ps = ips_dict[kinds]
        ps = ps.load(key, param)
        pss.append(ps)
    return pss


def _get_columns(df, columns, symbols=None, grouped_by_symbol=True):
    target_columns = []
    if columns is None:
        columns = df.columns
    if type(df.columns) == pd.MultiIndex:
        target_symbols = convert.get_symbols(df, grouped_by_symbol)
        if symbols is not None:
            target_symbols = list(set(target_symbols) & set(symbols))
        for i_columns in columns:
            if type(i_columns) is str:
                if grouped_by_symbol:
                    target_columns += [
                        (__symbol, i_columns) for __symbol in target_symbols
                    ]
                else:
                    target_columns += [
                        (i_columns, __symbol) for __symbol in target_symbols
                    ]
            elif isinstance(i_columns, Iterable) and len(i_columns) == 2:
                target_columns.append(i_columns)
            else:
                print(f"skip {i_columns} on ignore column process of minmax")
    else:
        target_columns = columns
        remaining_column = list(set(df.columns) - set(target_columns))
    return target_columns, remaining_column


def _concat_target_and_remain(original_df, processed_df, remaining_columns):
    original_columns = original_df.columns
    if remaining_columns is not None and len(remaining_columns) > 0:
        remaining_data = original_df[remaining_columns]
        data = pd.concat([processed_df, remaining_data], axis=1)
        # revert columns order to original
        data = data[original_columns]
    else:
        data = processed_df
    return data


class DiffPreProcess(ProcessBase):
    kinds = "Diff"

    def __init__(
        self,
        periods: int = 1,
        columns=None,
        key: str = None,
    ):
        if key is None:
            key = f"diff_{periods}"
        super().__init__(key)
        self.columns = columns
        self.periods = periods
        self.last_tick = None

    @property
    def option(self):
        return {"periods": self.periods, "columns": self.columns}

    @classmethod
    def load(self, key: str, params: dict):
        return DiffPreProcess(**params, key=key)

    def run(self, df: pd.DataFrame) -> dict:
        remaining_columns = None
        if self.columns is not None:
            target_columns, remaining_columns = _get_columns(df, self.columns)
            temp_data = df[target_columns]
        else:
            temp_data = df
        self.first_ticks = df.iloc[: self.periods]
        self.last_ticks = df.iloc[-self.periods :]
        temp_data = temp_data.diff(periods=self.periods)
        data = _concat_target_and_remain(df, temp_data, remaining_columns)
        return data

    def update(self, tick: pd.Series):
        """assuming data is previous result of run()

        Args:
            data (pd.DataFrame): previous result of run()
            tick (pd.Series): new row data
            option (Any, optional): Currently no option (Floor may be added later). Defaults to None.
        """
        new_data = tick - self.last_ticks.iloc[-self.periods]
        self.last_ticks = pd.concat([self.last_ticks[-self.periods + 1 :], tick])
        return new_data

    def get_minimum_required_length(self):
        return self.periods + 1

    def revert(self, data, base_values=None):
        columns = self.first_ticks.columns

        if isinstance(data, pd.DataFrame):
            available_columns = []
            for column in data.columns:
                if column in columns:
                    available_columns.append(column)

            if len(available_columns) > 0:
                if base_values is None:
                    base_values = self.first_ticks[available_columns]
                r_data = np.zeros_like(data)
                for start_index in range(self.periods):
                    temp_values = data[start_index :: self.periods].cumsum().fillna(0)
                    r_data[start_index :: self.periods] = (
                        temp_values + base_values.iloc[start_index]
                    )
                return pd.DataFrame(r_data, index=data.index, columns=available_columns)
            else:
                raise ValueError(
                    f"data has different columns: {data.columns} is not part of {columns}"
                )
        elif isinstance(data, np.ndarray):
            if len(data.shape) > 2:
                if base_values is None:
                    raise ValueError(
                        "base_value must be specified. Default is to revert entire data."
                    )
                axis = 0
            else:
                axis = 0
                if base_values is None:
                    base_values = self.first_ticks.values
                    if len(data.shape) == 2 and data.shape[1] != len(columns):
                        feature_size = data.shape[1]
                        if feature_size > len(columns):
                            raise ValueError(
                                f"can't determin column axis in the positions"
                            )
                        columns = columns[feature_size]
                        warnings.warn(
                            f"assume axis=1:{data.shape[1]} is a part of columns"
                        )

            r_data = np.zeros_like(data)
            for start_index in range(self.periods):
                temp_values = np.cumsum(data[start_index :: self.periods], axis=axis)
                temp_values = np.nan_to_num(temp_values)
                r_data[start_index :: self.periods] = temp_values + base_values
            return r_data

        else:
            raise TypeError(f"type {type(data)} is not supported.")


class LogPreProcess(ProcessBase):
    kinds = "Log"

    def __log(self, df: pd.DataFrame, columns: list):
        return df[columns].apply(np.log)

    def __exp(self, values):
        return np.exp(values)

    def __init__(self, columns: list = None, e=None):
        """Apply np.log for specified columns

        Args:
            columns (list, optional): target columns to apply np.log. Defaults to None and apply entire columns
            e (int, optional): base of log. Defaults to None and exp is used.
        """
        super().__init__("log")
        self.columns = columns
        if e is not None:
            log_base_value = np.log(e)
            if log_base_value == np.inf:
                log_base_value = np.log(float(e))
                self.__log = lambda df, columns: df[columns].apply(np.log) / np.log(
                    float(e)
                )
                self.__exp = lambda values: float(e) ** values
            else:
                self.__log = lambda df, columns: df[columns].apply(np.log) / np.log(e)
                self.__exp = lambda values: e**values

    def run(self, df: pd.DataFrame):
        target_columns, remaining_columns = _get_columns(df, self.columns)
        data = self.__log(df, target_columns)
        data = _concat_target_and_remain(df, data, remaining_columns)
        return data

    def revert(self, data):
        if isinstance(data, pd.DataFrame):
            target_columns, remaining_columns = _get_columns(data, self.columns)
            log_data = data[target_columns]
            r_data = self.__exp(log_data)
            r_data = _concat_target_and_remain(data, r_data, remaining_columns)
            return r_data
        return self.__exp(data)

    def get_minimum_required_length(self):
        return 1


class IDPreProcess(ProcessBase):
    kinds = "ID"

    def __init__(self, columns: list = None, decimals=None):
        """Convert Numeric values to ID (0 to X)

        Args:
            columns (list): _description_
            decimals(int|list, optional):
                When int is specified,
                if decimals > 0, Reduce digits by rounding the value. Ex.) round_digits=2 for 10324 become 103
                If decimals < 0, Recud decimal digits by rounding the value. Ex.) decimal_digits=3 for 3.14159265 become 3142:
                When list of int is specified, it should the same length as the columns.
                each columns are ID
        """
        super().__init__("id")
        if decimals is None or type(decimals) is int:
            self.decimals = decimals
        elif isinstance(decimals, Iterable):
            if columns is None or len(decimals) == len(columns):
                self.decimals = decimals
            else:
                raise ValueError("decimals must have same length as columns specified.")
        else:
            raise TypeError(
                f"decimans should be int or list. {type(decimals)} is specified."
            )
        self.columns = columns
        self.initialization_required = True

    def run(self, df: pd.DataFrame):
        org_columns = df.columns
        target_columns, remaining_columns = _get_columns(df, self.columns)
        temp_data = df[target_columns]
        if self.decimals is not None:
            if type(self.decimals) is int:
                temp_data = temp_data.round(self.decimals) * 10**-self.decimals
            else:
                temp_dfs = []
                for index in range(len(self.decimals)):
                    column = self.columns[index]
                    decimal = decimal[index]
                    temp_df = temp_data[column].round(decimal) * 10**-self.decimals
                    temp_dfs.append(temp_df)
                temp_data = pd.concat(temp_dfs, axis=1)
        id_df = temp_data + self.min_values.abs()
        id_df = id_df.astype(self.int_type)
        remaining_df = df[remaining_columns]
        df = pd.concat([id_df, remaining_df], axis=1)
        df = df[org_columns]
        return df

    def initialize(self, df: pd.DataFrame):
        # as run function would be called with partial data, caliculate min_values in advance for entire data.
        if self.columns is None:
            self.columns = df.columns
        params = {}
        df = df[self.columns]
        if self.decimals is not None:
            if type(self.decimals) is int:
                df = df.round(self.decimals) * 10**-self.decimals
            else:
                temp_dfs = []
                for index in range(len(self.decimals)):
                    decimal = decimal[index]
                    column = self.columns[index]
                    if decimal is not None:
                        temp_df = df[column].round(decimal) * 10**-self.decimals
                        temp_dfs.append(temp_df)
                    else:
                        temp_dfs.append(df[column])
                df = pd.concat(temp_dfs, axis=1)
        self.min_values = df.min()
        self.value_ranges = df.max() + self.min_values.abs()
        a_max_value = self.value_ranges.max().max()
        if a_max_value > 32768:
            self.int_type = "int16"
        elif a_max_value > 2147483648:
            self.int_type = "int32"
        else:
            self.int_type = "int64"
        self.initialization_required = False


class MultiFeatureIDPreProcess(IDPreProcess):
    kinds = "MFID"

    def __init__(self, columns: list, round_digits=None, decimal_digits=None):
        """Convert Numeric values to ID (0 to X) based on each columns
        Ex.) Data have MultiIndex[(USDJPY, Open), (USDJPY, Close),(USDJPY, Volume) (USDCHF, Open), (USDCHF, Close), (USDCHF, Volume)] as columns
        If [Open, Close] is specified, a minimum value is caliulated based on [Open, Close] values for each symbol. Then convert the values to IDs.
        Thus, Open ID and Close ID has same value for the symbol

        Args:
            columns (list): _description_
            round_digits(int, optional): Reduce digits by rounding the value. Ex.) round_digits=2 for 10324 become 103:
            decimal_digits(int, optional): Recud decimal digits by rounding the value. Ex.) decimal_digits=3 for 3.14159265 become 3142:
        """
        super().__init__("mfid")


class SimpleColumnDiffPreProcess(ProcessBase):
    kinds = "SCDiff"

    def __init__(
        self,
        base_column: str = "close",
        target_columns: list = ["open", "high", "low", "close"],
    ):
        """Simply caliculate diff: df[target_columns].iloc[1:] - df[base_column].iloc[:-1].values
        Assume base_column is close and target_columns is [open, high, low, close]

        Args:
            base_column (str, optional): Defaults to "close"
            target_columns (list, optional): Defaults to ["open", "high", "low", "close"].
        """
        super().__init__("scdiff")


class DailyCloseDiffPreProcess(ProcessBase):
    kinds = "DCDiff"

    def __init__(
        self, ohlc_columns=["open", "high", "low", "close"], last_datetime: time = None
    ):
        """Caliculate Diff based on last Close value of one day before


        Args:
            ohlc_columns (list, optional): _description_. Defaults to ["open", "high", "low", "close"].
        """
        super().__init__("dcdiff")

    def run(self, df):
        pass

    def revert(self, data):
        pass


class MinMaxPreProcess(ProcessBase):
    kinds = "MiniMax"

    def __init__(
        self,
        columns=None,
        scale=(-1, 1),
        min_values=None,
        max_values=None,
        key: str = "minmax",
    ):
        """Apply minimax for each column of data.
        Note that if params are not specified, mini max values are detected by data on running once only.
        So if data is partial data, mini max values will be not correct.

        Args:
            key (str, optional): identification of this process. Defaults to 'minmax'.
            scale (tuple, optional): minimax scale. Defaults to (-1, 1).
            min_value, max_value (dict, optional):  {{column_name: min/max value}}. Defaults to None and caliculate
            by provided data when run this process.
            columns (list, optional): specify column to ignore applying minimax or revert process. Defaults to []
        """
        super().__init__(key)
        if isinstance(scale, Iterable):
            if len(scale) == 2 and scale[0] < scale[1]:
                self.scale = scale
            else:
                raise ValueError(
                    "scale should have 2 elements with scale[0] < scale[1]"
                )
        else:
            self.scale = (-1, 1)

        if type(columns) is str:
            columns = [columns]
        if isinstance(columns, Iterable):
            columns = list(set(columns))
        elif columns is not None:
            raise TypeError("columns should be iterable")
        self.columns = columns

        self.initialization_required = True
        if min_values is not None:
            if max_values is not None:
                self.min_values = min_values
                self.max_values = max_values
                self.initialization_required = False
            else:
                raise ValueError("Both min and max values required for init.")
        elif max_values is not None:
            raise ValueError("Both min and max values required for init.")
        else:
            self.min_values = {}
            self.max_values = {}

    @classmethod
    def load(self, key: str, params: dict):
        option = {"scale": (-1, 1)}
        for k, value in params.items():
            if type(value) == list:
                option[k] = tuple(value)
            else:
                option[k] = value
        process = MinMaxPreProcess(key, **option)
        return process

    def initialize(self, data: pd.DataFrame):
        if self.columns is None:
            self.columns = data.columns
        self.run(data)
        self.initialization_required = False

    def run(
        self, data: pd.DataFrame, symbols: list = None, grouped_by_symbol=False
    ) -> dict:
        target_columns = _get_columns(data, self.columns, symbols, grouped_by_symbol)
        columns = list(set(data.columns) & set(target_columns))

        if len(self.min_values) > 0:
            _min = pd.Series(self.min_values)
            _max = pd.Series(self.max_values)
        else:
            _min = data[columns].min()
            self.min_values.update(_min.to_dict())
            _max = data[columns].max()
            self.max_values.update(_max.to_dict())

        _df, _, _ = standalization.mini_max(data[columns], _min, _max, self.scale)

        return _df

    def update(self, tick: pd.Series, do_update_minmax=True):
        result = {}

        for column in self.columns:
            new_value = tick[column]
            _min = self.min_values[column]
            _max = self.max_values[column]
            if do_update_minmax:
                if new_value < _min:
                    _min = new_value
                    self.min_values[column] = _min
                if new_value > _max:
                    _max = new_value
                    self.max_values[column] = _max

            scaled_new_value, _min, _max = standalization.mini_max(
                new_value, _min, _max, self.scale
            )
            result[column] = scaled_new_value

        new_data = pd.Series(result)
        return new_data

    def get_minimum_required_length(self):
        return 1

    def revert(self, data):
        """revert data minimaxed by this process

        Args:
            data (DataFrame|Series): log data

        Returns:
           reverted data. type is same as input
        """

        if isinstance(data, pd.DataFrame):
            data = data[self.columns]
            _min = pd.Series(self.min_values)
            _max = pd.Series(self.max_values)
            return standalization.revert_mini_max(data, _min, _max, self.scale)
        elif isinstance(data, pd.Series):
            column = data.name
            if column in self.columns:
                _min = self.min_values[column]
                _max = self.max_values[column]
            else:
                columns = data.index
                _min = []
                _max = []
                for column in columns:
                    if column in self.columns:
                        _min.append(self.min_values[column])
                        _max.append(self.max_values[column])
                _min = pd.Series(_min, index=columns)
                _max = pd.Series(_max, index=columns)
            return standalization.revert_mini_max_from_series(data, _min, _max, self.scale)
        else:
            print(f"type{data} is not supported")


class STDPreProcess(ProcessBase):
    kinds = "STD"

    def __init__(self):
        super().__init__("std")

    def run(self, df):
        return df

    def revert(self, data):
        return data


class PCTChange(ProcessBase):
    kinds = "PCTC"

    def __init__(self):
        super().__init__("pctc")

    def run(self, df):
        return df

    def revert(self, data):
        return data
