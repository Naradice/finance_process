import os
import sys
import unittest

import numpy as np
import pandas as pd

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)

from fprocess import preprocess


class TestPreProcess(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        self.org_data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "sample_data.csv"),
            index_col=0,
            parse_dates=True,
        )
        self.ohlc_columns = ["open", "high", "low", "close"]
        super().__init__(methodName)

    def test_diff_process(self):
        periods = 1
        dprorcess = preprocess.DiffPreProcess(periods=periods, columns=self.ohlc_columns)
        diff_data = dprorcess(self.org_data)
        diff_exp = self.org_data.diff(periods)
        self.assertEqual(len(self.org_data.columns), len(diff_data.columns))
        self.assertTrue((self.org_data.columns == diff_data.columns).all())

        sample_column = self.org_data.columns[0]
        for i in range(0, len(diff_exp)):
            process_value = diff_data[sample_column].iloc[i]
            exp_value = diff_exp[sample_column].iloc[i]
            if np.isnan(process_value) and np.isnan(exp_value):
                continue
            self.assertTrue(process_value == exp_value, f"{process_value} != {exp_value} on {i}")

    def test_diff_with_columns(self):
        periods = 1
        dprorcess = preprocess.DiffPreProcess(periods=periods, columns=["close"])
        diff_data = dprorcess(self.org_data)
        self.assertEqual(len(self.org_data.columns), len(diff_data.columns))
        self.assertTrue((self.org_data.columns == diff_data.columns).all())

    def test_revert_diff(self):
        periods = 1
        dprorcess = preprocess.DiffPreProcess(periods=periods)
        diff_data = dprorcess(self.org_data)
        r_data = dprorcess.revert(diff_data)
        sample_column = self.org_data.columns[0]
        for i in range(0, len(self.org_data)):
            process_value = r_data[sample_column].iloc[i]
            exp_value = self.org_data[sample_column].iloc[i]
            self.assertTrue(
                process_value == exp_value, f"{process_value} != {exp_value} on {i}"
            )

    def test_id_process(self):
        idprocess = preprocess.IDPreProcess(columns=["open", "close"])
        id_df = idprocess(self.org_data)
        print(id_df)

    def test_log_process(self):
        lprocess = preprocess.LogPreProcess(columns=self.ohlc_columns)
        log_data = lprocess(self.org_data)
        exp_data = self.org_data.apply(np.log)

        self.assertEqual(len(self.org_data.columns), len(log_data.columns))
        self.assertTrue((self.org_data.columns == log_data.columns).all())

        sample_column = self.org_data.columns[0]
        for i in range(0, len(exp_data)):
            process_value = log_data[sample_column].iloc[i]
            exp_value = exp_data[sample_column].iloc[i]
            if np.isnan(process_value) and np.isnan(exp_value):
                continue
            self.assertTrue(process_value == exp_value, f"{process_value} != {exp_value} on {i}")

    def test_revert_log(self):
        lprocess = preprocess.LogPreProcess(columns=self.ohlc_columns)
        log_data = lprocess(self.org_data)
        r_data = lprocess.revert(log_data)

        self.assertEqual(len(r_data.columns), len(log_data.columns))
        self.assertTrue((r_data.columns == log_data.columns).all())

        sample_column = self.ohlc_columns[0]
        for i in range(0, len(self.org_data)):
            process_value = r_data[sample_column].iloc[i]
            exp_value = self.org_data[sample_column].iloc[i]
            if np.isnan(process_value) and np.isnan(exp_value):
                continue
            self.assertAlmostEqual(process_value, exp_value, msg=f"{process_value} != {exp_value} on {i}")

    def test_minmax_process(self):
        pass

    def test_save_processes(self):
        file_name = "./preprocess.json"
        dprorcess = preprocess.DiffPreProcess(periods=1, columns=self.ohlc_columns)
        dprorcess2 = preprocess.DiffPreProcess(periods=2, columns=["volume"])
        processes = [dprorcess, dprorcess2]
        preprocess.save_preprocesses(processes)
        self.assertTrue(os.path.exists(file_name))
        loaded_processes = preprocess.load_preprocess(file_name)
        self.assertTrue(processes == loaded_processes)


if __name__ == "__main__":
    unittest.main()
