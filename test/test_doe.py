import pandas as pd
import numpy as np

import pytest
from pytest import mark

from doe import create_fact_plan


class TestDOE:
    def test_nominal(self):

        # 2x2 case
        data_2x2_in = pd.DataFrame(data=np.array([[0, 10],
                                                  [1, 20]]),
                                   index=["min", "max"],
                                   columns=["feat1", "feat2"]
                                   )

        data_2x2_plan = pd.DataFrame(data=np.array([[0, 10],
                                                    [1, 10],
                                                    [0, 20],
                                                    [1, 20]]),
                                     index=[0, 1, 2, 3],
                                     columns=["feat1", "feat2"]
                                     )

        try:
            pd.testing.assert_frame_equal(create_fact_plan(data_2x2_in), data_2x2_plan)
        except AssertionError:
            pytest.fail("Test failed on 2x2 nominal case.")

        # 3x2 case
        data_3x2_in = pd.DataFrame(data=np.array([[0, 10, 50],
                                                  [1, 20, 60]]),
                                   index=["min", "max"],
                                   columns=["feat1", "feat2", "feat3"]
                                   )

        data_3x2_plan = pd.DataFrame(data=np.array([[0, 10, 50],
                                                    [1, 10, 50],
                                                    [0, 20, 50],
                                                    [1, 20, 50],
                                                    [0, 10, 60],
                                                    [1, 10, 60],
                                                    [0, 20, 60],
                                                    [1, 20, 60]]),
                                     index=[i for i in range(8)],
                                     columns=["feat1", "feat2", "feat3"]
                                     )
        try:
            pd.testing.assert_frame_equal(create_fact_plan(data_3x2_in), data_3x2_plan)
        except AssertionError:
            pytest.fail("Test failed on 3x2 nominal case.")

    def test_NaN(self):

        data_NaN_in = pd.DataFrame(data=np.array([[0, np.nan],
                                                  [1, 20]]),
                                   index=["min", "max"],
                                   columns=["feat1", "feat2"]
                                   )

        data_NaN_plan = pd.DataFrame(data=np.array([[0, np.nan],
                                                    [1, np.nan],
                                                    [0, 20],
                                                    [1, 20]]),
                                     index=[0, 1, 2, 3],
                                     columns=["feat1", "feat2"]
                                     )

        try:
            pd.testing.assert_frame_equal(create_fact_plan(data_NaN_in), data_NaN_plan)
        except AssertionError:
            pytest.fail("Test failed on 2x2 with NaN case.")

    def test_str(self):

        data_str_in = pd.DataFrame(data=np.array([["min", 10],
                                                  ["max", 20]]),
                                   index=["min", "max"],
                                   columns=["feat1", "feat2"]
                                   )

        data_str_plan = pd.DataFrame(data=np.array([["min", 10],
                                                    ["max", 10],
                                                    ["min", 20],
                                                    ["max", 20]]),
                                     index=[0, 1, 2, 3],
                                     columns=["feat1", "feat2"]
                                     )

        try:
            pd.testing.assert_frame_equal(create_fact_plan(data_str_in), data_str_plan)
        except AssertionError:
            pytest.fail("Test failed on 2x2 with string case.")

    def test_nominal_gsd(self):

        # 2x2 GSD (red default) case
        data_2x2_in = pd.DataFrame(data=np.array([[0, 10],
                                                  [1, 20]]),
                                   index=["min", "max"],
                                   columns=["feat1", "feat2"]
                                   )

        data_2x2_gsd_plan = pd.DataFrame(data=np.array([[0, 10],
                                                        [1, 20]]),
                                         index=[0, 1],
                                         columns=["feat1", "feat2"]
                                         )

        try:
            pd.testing.assert_frame_equal(create_fact_plan(data_2x2_in, method_choice="gsd"), data_2x2_gsd_plan)
        except AssertionError:
            pytest.fail("Test failed on 2x2 GSD (default reduction) nominal case.")

        # 3x2 GSD (red: 3) case
        data_3x2_in = pd.DataFrame(data=np.array([[0, 10, 50],
                                                  [1, 20, 60]]),
                                   index=["min", "max"],
                                   columns=["feat1", "feat2", "feat3"]
                                   )

        data_3x2_gsd_plan = pd.DataFrame(data=np.array([[0, 10, 50],
                                                        [0, 20, 60],
                                                        [1, 10, 60]]),
                                         index=[i for i in range(3)],
                                         columns=["feat1", "feat2", "feat3"]
                                         )
        try:
            pd.testing.assert_frame_equal(create_fact_plan(data_3x2_in,
                                                           method_choice="gsd",
                                                           reduction_ratio=3),
                                          data_3x2_gsd_plan)
        except AssertionError:
            pytest.fail("Test failed on 3x2 GSD (reduction: 3) nominal case.")

    def test_wrong_type_df(self):

        data_2x2_in = "DataFrame becomes a string."

        with pytest.raises(AttributeError):
            create_fact_plan(data_2x2_in)

    def test_unknown_method(self):

        data_2x2_in = pd.DataFrame(data=np.array([[0, 10],
                                                  [1, 20]]),
                                   index=["min", "max"],
                                   columns=["feat1", "feat2"]
                                   )

        with pytest.raises(AssertionError):
            create_fact_plan(data_2x2_in, method_choice="foo")

    def test_red_ratio_type(self):

        data_2x2_in = pd.DataFrame(data=np.array([[0, 10],
                                                  [1, 20]]),
                                   index=["min", "max"],
                                   columns=["feat1", "feat2"]
                                   )

        with pytest.raises(AssertionError):
            create_fact_plan(data_2x2_in, method_choice="gsd", reduction_ratio="not_an_integer")
