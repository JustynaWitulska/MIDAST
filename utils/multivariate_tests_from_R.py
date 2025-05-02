from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()
import os
import sys

sys.path.append("../")
path_to_R = os.getenv("R_PATH")
if not path_to_R:
    raise ValueError("Environment variable 'R_PATH' is not set or is empty.")
os.environ["R_HOME"] = path_to_R
path = os.path.join(path_to_R, "bin/x64/")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(path)
else:
    os.environ["PATH"] = f"{path};" + os.environ["PATH"]
path2 = os.path.join(path_to_R, "bin")
# os.environ["path"] += f";{path2};"

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# # Suppress R warning/info messages
# rpy2_logger.setLevel(logging.ERROR)


# Optional: suppress R stdout temporarily
@contextmanager
def suppress_r_output():
    from rpy2.rinterface_lib import callbacks

    original_writeconsole = callbacks.consolewrite_print
    callbacks.consolewrite_print = lambda x: None
    try:
        yield
    finally:
        callbacks.consolewrite_print = original_writeconsole


pandas2ri.activate()

import sys

sys.path.append("..")
import pandas as pd

pd.DataFrame.iteritems = pd.DataFrame.items
import numpy as np

utils_r = importr("utils")
base = importr("base")

utils_r.chooseCRANmirror(ind=1)

np_r = importr("np")
kernel_two_sample_test = importr("maotai")
cramer_test = importr("cramer")
copula_based_test = importr("TwoCop")


class KernelDensitiesTest:
    def __init__(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
    ) -> None:
        """
        Initialize the KernelDensitiesTest with two dataframes.

        Args:
            df1 (pd.DataFrame): First dataset.
            df2 (pd.DataFrame): Second dataset.
        """
        self.x = df1
        self.y = df2

    def conduct_test(self, boot_num: int) -> tuple[float, float]:
        """
        Conduct the kernel densities test.

        Args:
            boot_num (int): Number of bootstrap samples.

        Returns:
            tuple[float, float]: p-value and test statistic.
        """
        with suppress_r_output():
            results = np_r.npdeneqtest(self.x, self.y, boot_num=boot_num)
        results2dict = dict(zip(results.names, list(results)))
        return results2dict["Tn.P"][0], results2dict["Tn"][0]


class MMDTest:
    def __init__(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
    ) -> None:
        """
        Initialize the MMDTest with two dataframes.

        Args:
            df1 (pd.DataFrame): First dataset.
            df2 (pd.DataFrame): Second dataset.
        """
        self.x = df1
        self.y = df2

    def conduct_test(self) -> tuple[float, float]:
        """
        Conduct the Maximum Mean Discrepancy (MMD) test.

        Returns:
            tuple[float, float]: p-value and test statistic.
        """
        ro.r(
            """
                # create a function `f`
                f <- function(dat1, dat2, lab) {
                    dmat <- as.matrix(dist(rbind(dat1, dat2)))
                    kmat <- exp(-(dmat^2)) 
                    result <- mmd2test(kmat, lab)
                    pvalue <- result$p.value
                    statistic <- result$statistic
                    return(c(pvalue, statistic))
                }
            """
        )
        lab = np.array([1] * self.x.shape[0] + [2] * self.y.shape[0])
        kernel_two_sample_test = ro.globalenv["f"]
        results = kernel_two_sample_test(self.x, self.y, lab)
        return results[0], results[1]


class CramerTest:
    def __init__(
        self,
        values1: np.ndarray,
        values2: np.ndarray,
    ) -> None:
        """
        Initialize the CramerTest with two arrays.

        Args:
            values1 (np.ndarray): First dataset.
            values2 (np.ndarray): Second dataset.
        """
        self.x = values1
        self.y = values2

    def conduct_test(self, nboot: int = 1000, kernel: str = "phiLog") -> tuple[float, float]:
        """
        Conduct the Cramer test.

        Args:
            nboot (int): Number of bootstrap samples. Default is 1000.
            kernel (str): Kernel type. Default is "phiLog".

        Returns:
            tuple[float, float]: p-value and test statistic.
        """
        ro.r(
            """
                # create a function `f`
                f <- function(x, y, replicates, kernel) {
                    result <- cramer.test(x,y,replicates=replicates, kernel=kernel)
                    pvalue <- result$p.value
                    statistic <- result$statistic
                    return(c(pvalue, statistic))
                }
            """
        )
        cramer_two_sample_test = ro.globalenv["f"]
        results = cramer_two_sample_test(self.x, self.y, nboot, kernel)
        return results[0], results[1]


class CopulaTest:
    def __init__(
        self,
        values1: np.ndarray,
        values2: np.ndarray,
    ) -> None:
        """
        Initialize the CopulaTest with two arrays.

        Args:
            values1 (np.ndarray): First dataset.
            values2 (np.ndarray): Second dataset.
        """
        self.x = values1
        self.y = values2

    def conduct_test(self, boot_num: int) -> tuple[float, float]:
        """
        Conduct the copula-based test.

        Args:
            boot_num (int): Number of bootstrap samples.

        Returns:
            tuple[float, float]: p-value and test statistic.
        """
        results = copula_based_test.TwoCop(self.x, self.y, Nsim=boot_num)
        results2dict = dict(zip(results.names, list(results)))
        return results2dict["pvalue"][0], results2dict["cvm"][0]
