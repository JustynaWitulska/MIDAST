from dotenv import load_dotenv
load_dotenv()

import json
import multiprocessing as mp
import os
import time
import warnings
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.multivariate_statistical_test_method import ChangeDetector

USE_R_TEST: Union[str, None] = os.getenv("USE_R_TESTS")
if USE_R_TEST:
    from utils.multivariate_tests_from_R import KernelDensitiesTest, MMDTest
from utils.ks_2samp import ks_2samp
from utils.ndtest import ks2d2s

def change_point_detection(
    df: pd.DataFrame,
    method: str,
    n_bkps: Union[int, None],
    window_size: int,
    shift: int,
    alpha: float = 0.05,
    kernel: str = "phiFracA",
    bootstrap_num: int | None = None
) -> List[int]:
    """
    Detects change points in the given DataFrame using the specified method.

    Args:
        df (pd.DataFrame): Input data.
        method (str): Statistical test method to use.
        n_bkps (Union[int, None]): Maximum number of change points to detect.
        window_size (int): Size of the sliding window.
        shift (int): Shift size for the sliding window.
        alpha (float): Significance level for the test.
        kernel (str): Kernel type for specific tests (if CramerTest is used).
        bootstrap_num (int): Number of Bootstrap replications (needed for some tests for critical intervals estimation).

    Returns:
        List[int]: List of detected change points.
    """
    shift_group: int = 10

    if isinstance(bootstrap_num, type(None)):
        bn = 50 if method == "KernelDensitiesTest" else 200
    else:
        bn = bootstrap_num

    if method in ["KSTest", "MMDTest", "TopologyTest", "CopulaTest", "KernelDensitiesTest"]:
        change_detector_instance = ChangeDetector(test_name=method, bn=bn)
        results_df = change_detector_instance.fit(df=df.values, window_size=window_size, shift=shift)

        results = change_detector_instance.analyze_results(
            results_df=results_df,
            alpha=alpha,
            max_no_changes=n_bkps,
            shift_group=shift_group,
        )

    elif method == "CramerTest":
        change_detector_instance = ChangeDetector(test_name=method, bn=bn)

        results_df = change_detector_instance.fit(
            df=df.values,
            window_size=window_size,
            shift=shift,
            kernel=kernel,
        )

        results = change_detector_instance.analyze_results(
            results_df=results_df,
            alpha=alpha,
            max_no_changes=n_bkps,
            shift_group=shift_group, 
        )

    else:
        raise NotImplementedError(f"Method {method} is not implemented.")
    return results


def double_check(
    df: pd.DataFrame,
    cp_detected: Union[List[int], None],
    alpha: float,
    test_name: str,
    window_size_original: int,
    window_size_checking: int = 500,
    bootstrap_num: int | None = None,
) -> Tuple[Union[List[int], str], Union[List[int], str]]:
    """
    Double-checks detected change points for statistical significance.

    Args:
        df (pd.DataFrame): Input data.
        cp_detected (Union[List[int], None]): List of detected change points.
        alpha (float): Significance level for the test.
        test_name (str): Statistical test method to use.
        window_size_original (int): Original window size used for detection.
        window_size_checking (int): Window size for double-checking.
        bootstrap_num (int): Number of Bootstrap replications (needed for some tests for critical intervals estimation).

    Returns:
        Tuple[Union[List[int], str], Union[List[int], str]]: Filtered change points and out-of-testing change points.
    """
    dimension: int = df.shape[1]

    cp_filtered: Union[List[int], str] = []
    cp_out_of_testing: Union[List[int], str] = []

    if isinstance(bootstrap_num, type(None)):
        bootstrap_num = 50 if method == "KernelDensitiesTest" else 200

    if cp_detected is not None:
        for cp in cp_detected:
            begin: int = max(0, cp - window_size_checking)
            end: int = min(len(df), cp + window_size_checking)

            if (cp - begin) <= window_size_original or (end - cp) <= window_size_original:
                cp_out_of_testing.append(cp)
            else:
                min_segment_length: int = min([cp - begin, end - cp])

                begin = max(0, cp - min_segment_length)
                end = min(len(df), cp + min_segment_length)

                values1 = df.iloc[begin:cp].values
                values2 = df.iloc[cp:end].values

                if test_name == "KSTest":
                    if dimension == 2:
                        x1, y1 = values1.T
                        x2, y2 = values2.T
                        result = ks2d2s(x1, y1, x2, y2, extra=True)
                    else:
                        stat, _, _, pvalue = ks_2samp(x_val=values1, y_val=values2, alpha=alpha)
                        result = [pvalue, stat]

                elif test_name == "KernelDensitiesTest":
                    if USE_R_TEST:
                        test_instance = KernelDensitiesTest(
                            df1=pd.DataFrame(values1),
                            df2=pd.DataFrame(values2),
                        )
                        result = test_instance.conduct_test(boot_num=bootstrap_num)
                    else:
                        raise ValueError(
                            "USE_R_TEST set as False. Please download and configure R-dependencies firstly, change R_PATH in .env set USE_R_TEST as True - if you want to use KernelDensitiesTest, MMDTest or Cramer-von-Mises test."
                        )
                elif test_name == "MMDTest":
                    if USE_R_TEST:
                        test_instance = MMDTest(
                            df1=pd.DataFrame(values1),
                            df2=pd.DataFrame(values2),
                        )
                        result = test_instance.conduct_test()
                else:
                    cp_filtered, cp_out_of_testing = "not implemented", "not implemented"

                if result[0] < alpha:
                    cp_filtered.append(cp)

    return cp_filtered, cp_out_of_testing
