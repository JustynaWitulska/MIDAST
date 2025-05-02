from dotenv import load_dotenv
load_dotenv()

import json
import multiprocessing as mp
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


from src.multivariate_statistical_test_method import ChangeDetector

USE_R_TEST = os.getenv("USE_R_TESTS")
if USE_R_TEST:
    from utils.multivariate_tests_from_R import KernelDensitiesTest, MMDTest
from utils.ks_2samp import ks_2samp
from utils.ndtest import ks2d2s

MC_TRIALS = 100
N_BKPS = None
ALPHA = 0.05
N_LIST = [1000]
W_LIST = [200]


def change_point_detection(
    df: pd.DataFrame,
    method: str,
    n_bkps: int,
    window_size: int,
    shift: int,
    alpha: float = 0.05,
    kernel: str = "phiFracA",
):
    shift_group = 10

    if (method == "KSTest") or (method == "MMDTest") or (method == "TopologyTest") or (method == "CopulaTest"):
        change_detector_instance = ChangeDetector(test_name=method)
        results_df = change_detector_instance.fit(df=df.values, window_size=window_size, shift=shift)

        results = change_detector_instance.analyze_results(
            results_df=results_df,
            alpha=alpha,
            max_no_changes=n_bkps,
            shift_group=shift_group,
        )

    elif method == "KernelDensitiesTest":
        change_detector_instance = ChangeDetector(test_name=method, bn=50)
        results_df = change_detector_instance.fit(df=df.values, window_size=window_size, shift=shift)

        results = change_detector_instance.analyze_results(
            results_df=results_df,
            alpha=alpha,
            max_no_changes=n_bkps,
            shift_group=shift_group,
        )

    elif method == "CramerTest":
        change_detector_instance = ChangeDetector(test_name=method)

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
        raise (NotImplementedError)
    return results


def double_check(df, cp_detected, alpha, test_name, window_size_original, window_size_checking=500):
    dimension = df.shape[1]

    cp_filtered = []
    cp_out_of_testing = []


    if not isinstance(cp_detected, type(None)):

        for cp in cp_detected:

            begin = max(0, cp - window_size_checking)
            end = min(len(df), cp + window_size_checking)

            if (cp-begin) <= window_size_original or (end - cp) <= window_size_original:
                cp_out_of_testing.append(cp)

            else:

                min_segment_length = min([cp-begin, end-cp])

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
                        stat, _, _, pvalue = ks_2samp(x_val=values1, y_val=values2, alpha=0.05)  # alpha=self.alpha)
                        result = [pvalue, stat]

                elif test_name == "KernelDensitiesTest":
                    if USE_R_TEST:
                        test_instance = KernelDensitiesTest(
                            df1=pd.DataFrame(values1),
                            df2=pd.DataFrame(values2),
                        )
                        result = test_instance.conduct_test(boot_num=50)
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
                    raise (NotImplementedError)

                if result[0] < alpha:
                    cp_filtered.append(cp)

    return cp_filtered, cp_out_of_testing