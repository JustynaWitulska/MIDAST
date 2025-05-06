# Welcome to MIDAST - `M`ult`I`dimensional `DA`ta `S`egmentation based on `T`wo-sample test 

## Introduction
This repository implements MIDAST that is a novel approach for segmenting multivariate data efficiently. It enhances accuracy and interpretability by leveraging two-sample statistical tests for equality of distribution. The core of MIDAST is available in src/multivariate_statistical_test_method.py. The repository contain also additional files necessary to test baseline methods (e.g., e-Divisive used in utils/cp_detection_R.py) and make simulations (all files with suffix `_simulations.py` from src/).

![idea3 (1)](https://github.com/user-attachments/assets/12f190f4-4030-43ec-8c0c-b2cebbb14720)


## Features
- Efficient segmentation of high-dimensional data
- Handles both linear and non-linear relationships
- Scalable to large datasets
- Easy integration with existing workflows

## Installation
Ensure you have >= Python 3.8 installed. Install dependencies using:

```bash
pip install -r requirements.txt
```

### Environment Configuration

In the root directory of the repository, duplicate the `.env.sample` file and rename the copy to `.env` (remove the `.sample` suffix).  
This setup ensures that sensitive credential files are not tracked by version control and are not accidentally committed to the repository.  
The `.env.sample` file contains example values to guide you in setting up your own environment variables.


## Usage
The examples of using is given in `examples/test.py`. Minimal working example is as follows:

```python
from utils.stblrnd import sub_gaussian_vect_with_corr_change_v2
from src.multivariate_statistical_test_method import ChangeDetector
import pandas as pd

# set sample details
n = 1000 # number of observations
n_star = 500 # change point
d = 2 # dimenson

# set parameters of the sub-gaussian vector (taken as example data)
par_alpha = par_alpha2 = 1.5
rho_before = -0.9
rho_after = 0.5

# genarate data
vect = sub_gaussian_vect_with_corr_change_v2(alpha=par_alpha, d=d, n=n, n_star=n_star, 
                                    rho_before=rho_before, rho_after=rho_after, alpha2=par_alpha2)
df = pd.DataFrame(vect).rename(columns={0: "x", 1: "y"}).reset_index()

# set parameters of the MIDAST
test_name = "KSTest"
window_size = 200
shift = 10
n_bkps = 1
shift_group = 10

# initialize ChangeDetector
change_detector_instance = ChangeDetector(test_name=test_name)

# find change points
results_df = change_detector_instance.fit(
    df=df, window_size=window_size, shift=shift)

change_points = change_detector_instance.analyze_results(
    results_df=results_df, alpha=alpha, max_no_changes=n_bkps, shift_group=shift_group,
)

```



---

## Parameters & Configuration
- test_name -- name of the two-sample statistical test to use,
- window_size -- corresponds to the length of the samples taken for two-sample test for equality of multivariate distribution,
- shift -- is set to reduce the set of possible change point indexes. It causes the algorithm to take all s observations for testing; predicted change points can only be a multiple of shift.
- alpha -- the significance level below which we reject the null hypothesis in testing the equality of multivariate distributions,
- shift_group (k) -- determines if detected change points belong to the same group. Two consecutive change points i, j belong to the same group if |i-j|<k. 
- n_bkps -- number of change points. It's optional parameter that describes the expected maximum number of change points. If it is None, then the algorithm returns all points for which the tests for equality of distribution rejected the null hypothesis (sorted from the most probable to the less probable).
- max_pvalues_for_grouping (optional) - maximum number of p-values to consider for grouping.
- bn (optional) -- Number of bootstrap samples (necessary for some of two-sample tests),
- based_on - criterion for selecting change points, either "statistic" or "pvalue". Default is "statistic".
- other parameters (**kwargs) can be connected with selected test_name (e.g., for `CramerTest` we can additionaly select kernel. For more details, see source code).

The `test_name` for tests used by MIDAST available in the current version are as follows:
- `KSTest` (from https://github.com/syrte/ndtest for d=2, and from https://github.com/o-laurent/multivariate-ks-test for d>2),
- `KSTest_DKW` (from https://github.com/o-laurent/multivariate-ks-test for all dimensions d),
- `TopologyTest` (from https://github.com/dioscuri-tda/topotests)
- `KernelDensitiesTest` (from https://www.rdocumentation.org/packages/np/versions/0.60-18/topics/npdeneqtest)
- `MMDTest` (from https://rdrr.io/cran/maotai/src/R/mmd2test.R)
- `CramerTest` (documentation: https://cran.r-project.org/web//packages//cramer/cramer.pdf)
- `CopulaTest` (from: https://www.rdocumentation.org/packages/TwoCop/versions/1.0/topics/TwoCop)

Only `KSTest` and `TopologyTest` are implemented in pure Python. The remaining tests are implemented in R and they require to set `USE_R_TESTS` as True and provide `R_PATH` in the `.env` file. If you would like to use only Python packages, set `USE_R_TESTS` as False.


Guidelines for parameterization can be found in the paper: <TBU>.

---

# Reference

If you find this code useful, you may cite the following paper:

<TBU>
```latex
@article{...,
  title={...},
  author={...},
  journal={...},
  year={...},
}
```

## Sources
[1]
[2]

