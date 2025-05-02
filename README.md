# Welcome to MIDAST - `M`ult`I`dimensional `DA`ta `S`egmentation based on `T`wo-sample test 

## Introduction
MIDAST is a novel approach for segmenting multivariate data efficiently. It enhances accuracy and interpretability by leveraging []. This method is particularly useful for applications in []. 


![idea3 (1)](https://github.com/user-attachments/assets/12f190f4-4030-43ec-8c0c-b2cebbb14720)


## Features
- Efficient segmentation of high-dimensional data
- Handles both linear and non-linear relationships
- Scalable to large datasets
- Easy integration with existing workflows

## Installation & Configuration
Ensure you have >= Python 3.8 installed. Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
The examples of using is given in `main.py`. Minimal working example is as follows:

```python
from utils.stblrnd import sub_gaussian_vect_with_corr_change_v2
from src.multivariate_statistical_test_method import ChangeDetector
import pandas as pd

alpha = alpha2 = 1.5
n = 1000
n_star = 500
rho_before = -0.9
rho_after = 0.5
test_name = "KSTest"

vect = sub_gaussian_vect_with_corr_change_v2(alpha=alpha, d=2, n=n, n_star=n_star, 
                                    rho_before=rho_before, rho_after=rho_after, alpha2=alpha2)

df = pd.DataFrame(vect).rename(columns={0: "x", 1: "y"}).reset_index()


window_size = 200
shift = 10
n_bkps = 1
shift_group = 10

change_detector_instance = ChangeDetector(test_name=test_name)
results_df = change_detector_instance.fit(
    df=df, window_size=window_size, shift=shift, kernel="phiFracA",
)
change_points = change_detector_instance.analyze_results(
    results_df=results_df, alpha=alpha, max_no_changes=n_bkps, shift_group=shift_group,
)

```



---

## Parameters & Configuration
TBA


---

## Examples & Visualization
Include examples and visualizations if applicable.









