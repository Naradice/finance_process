import os
import sys
import unittest

import pandas as pd

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)

from fprocess import standalization as std


class TestPreProcess(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_diff_process(self):
        pass


if __name__ == "__main__":
    unittest.main()
