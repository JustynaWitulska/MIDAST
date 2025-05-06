
import sys
sys.path.append("..")

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
alpha = 0.05

# initialize ChangeDetector
change_detector_instance = ChangeDetector(test_name=test_name)

# find change points
results_df = change_detector_instance.fit(
    df=df.values, window_size=window_size, shift=shift)

change_points = change_detector_instance.analyze_results(
    results_df=results_df, alpha=alpha, max_no_changes=n_bkps, shift_group=shift_group,
)

print(change_points)