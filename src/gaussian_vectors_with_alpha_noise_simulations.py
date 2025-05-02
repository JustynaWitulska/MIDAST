import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

import multiprocessing as mp
import os
from typing import Dict, List

from utils.stblrnd import gaussian_vect_with_stable_noise_corr_change, gaussian_vect_with_stable_noise_corr_change_case3


class GaussianVectorWithStableNoise:
    def __init__(self) -> None:
        """
        Initialize the GaussianVectorWithStableNoise class.
        """
        self.setup: Dict[int, Dict] = {}

    def set_setup(self, setup: Dict[int, Dict]) -> None:
        """
        Set the experimental setup.

        Parameters:
        - setup (Dict[int, Dict]): A dictionary where keys are experiment IDs and values are setup configurations.
        """
        self.setup = setup

    def single_trial(
        self,
        experiment_id: int,
        n: int,
        n_star: int,
        save_dir: str,
        save_fig: bool = False,
    ) -> None:
        """
        Perform a single trial simulation for a given experiment.

        Parameters:
        - experiment_id (int): The ID of the experiment.
        - n (int): Total number of samples.
        - n_star (int): Number of samples before the correlation change.
        - save_dir (str): Directory to save the results.
        - save_fig (bool): Whether to save the figures. Defaults to False.
        """
        setup = self.setup.get(experiment_id)
        if setup:
            p = setup["p"]
            alpha = setup["alpha"]
            rho1 = setup["rho_before"]
            rho2 = setup["rho_after"]

            vect = gaussian_vect_with_stable_noise_corr_change(
                alpha=alpha, rho1=rho1, rho2=rho2, n=n, n_star=n_star, p=p
            )

            df = pd.DataFrame(vect).rename(columns={0: "x", 1: "y"})
            df["rho"] = [rho1] * n_star + [rho2] * (df.shape[0] - n_star)
            df["color"] = df.rho.apply(lambda x: "red" if x == rho1 else "blue")
            name = f"Simulation_n_{n}_alpha_{alpha}_p_{p}_rho1_{rho1}_rho2_{rho2}"

            if not os.path.exists(f"./{save_dir}"):
                os.makedirs(f"./{save_dir}")

            df.to_csv(f"{save_dir}/{name}.csv")

            if save_fig:
                if not os.path.exists(f"./{save_dir}/Figures"):
                    os.makedirs(f"./{save_dir}/Figures")

                plt.figure()
                ax = plt.axes(projection="3d")
                for color in df.color.unique():
                    ax.plot3D(
                        df[df.color == color].x,
                        df[df.color == color].y,
                        df[df.color == color].index,
                        ".",
                        color=color,
                    )
                    plt.xlabel("x")
                    plt.ylabel("y")
                    ax.set_zlabel("index")
                plt.savefig(f"{save_dir}/Figures/{name}_3d_scatter.png")
                plt.close()

                sns.jointplot(data=df[["x", "y", "rho"]], x="x", y="y", hue="rho", kind="kde")
                plt.savefig(f"{save_dir}/Figures/{name}_2densities.png")
                plt.close()

                sns.kdeplot(df[["x", "y", "rho"]], x="x", hue="rho")
                plt.savefig(f"{save_dir}/Figures/{name}_density_x.png")
                plt.close()

                sns.kdeplot(df[["x", "y", "rho"]], x="y", hue="rho")
                plt.savefig(f"{save_dir}/Figures/{name}_density_y.png")
                plt.close()

    def single_trial_case3(self, experiment_id: int, n: int, save_dir: str, save_fig: bool = False) -> None:
        """
        Perform a single trial simulation for case 3.

        Parameters:
        - experiment_id (int): The ID of the experiment.
        - n (int): Total number of samples.
        - save_dir (str): Directory to save the results.
        - save_fig (bool): Whether to save the figures. Defaults to False.
        """
        setup = self.setup.get(experiment_id)
        if setup:
            p = setup["p"]
            alpha = setup["alpha"]
            rho1 = setup["rho1"]
            rho2 = setup["rho2"]
            n1 = int(0.33 * n)
            n2 = int(0.66 * n)
            n3 = n - n1 - n2
            vect = gaussian_vect_with_stable_noise_corr_change_case3(
                alpha=alpha, rho1=rho1, rho2=rho2, n1=n1, n2=n2, n3=n3, p=p
            )

            df = pd.DataFrame(vect).rename(columns={0: "x", 1: "y"})
            df["rho"] = [rho1] * n1 + [rho2] * n2 + [rho1] * (n - n2 - n1)
            df["color"] = df["rho"].replace(rho1, "green").replace(rho2, "red").values
            name = f"Simulation_n_{n}_alpha_{alpha}_rho1_{rho1}_rho2_{rho2}_rho3_{rho1}"

            if not os.path.exists(f"./{save_dir}"):
                os.makedirs(f"./{save_dir}")

            df.to_csv(f"{save_dir}/{name}.csv")

            if save_fig:
                if not os.path.exists(f"./{save_dir}/Figures"):
                    os.makedirs(f"./{save_dir}/Figures")

                plt.figure()
                ax = plt.axes(projection="3d")
                for color in df.color.unique():
                    ax.plot3D(
                        df[df.color == color].x,
                        df[df.color == color].y,
                        df[df.color == color].index,
                        ".",
                        color=color,
                    )
                    plt.xlabel("x")
                    plt.ylabel("y")
                    ax.set_zlabel("index")
                plt.savefig(f"{save_dir}/Figures/{name}_3d_scatter.png")
                plt.close()

                sns.jointplot(data=df[["x", "y", "rho"]], x="x", y="y", hue="rho", kind="kde")
                plt.savefig(f"{save_dir}/Figures/{name}_2densities.png")
                plt.close()

                sns.kdeplot(df[["x", "y", "rho"]], x="x", hue="rho")
                plt.savefig(f"{save_dir}/Figures/{name}_density_x.png")
                plt.close()

                sns.kdeplot(df[["x", "y", "rho"]], x="y", hue="rho")
                plt.savefig(f"{save_dir}/Figures/{name}_density_y.png")
                plt.close()

    def make_simulations(self, setup: Dict[int, Dict], n_list: List[int], case: int = 1) -> None:
        """
        Generate simulations for multiple experiments using multiprocessing.

        Parameters:
        - setup (Dict[int, Dict]): A dictionary where keys are experiment IDs and values are setup configurations.
        - n_list (List[int]): List of sample sizes for simulations. Defaults to [200, 1000, 2000].
        - case (int): Case identifier for the simulation. Defaults to 1.
        """
        self.setup = setup
        for n in n_list:
            for coeff in [0.3, 0.5]:
                n_star = int(coeff * n)
                save_dir = f"./case_2_p1/Simulations_n_{n}_change_{n_star}"

                # Step 1: Init multiprocessing.Pool()
                pool = mp.Pool(mp.cpu_count())

                # Step 2: `pool.apply` the `single_trial()`
                _ = [
                    pool.apply(
                        self.single_trial,
                        args=(experiment_id, n, n_star, save_dir, False),
                    )
                    for experiment_id in setup
                ]

                # Step 3: Don't forget to close
                pool.close()

    def make_simulations_case3(self, setup: Dict[int, Dict], n_list: List[int]) -> None:
        """
        Generate simulations for case 3 using multiprocessing.

        Parameters:
        - setup (Dict[int, Dict]): A dictionary where keys are experiment IDs and values are setup configurations.
        - n_list (List[int]): List of sample sizes for simulations. Defaults to [300, 1000, 2000].
        """
        self.setup = setup
        for n in n_list:
            n1 = int(0.33 * n)
            n2 = int(0.66 * n)
            save_dir = f"./case_4_p1/Simulations_n_{n}_{n1}_{n2}"

            # Step 1: Init multiprocessing.Pool()
            pool = mp.Pool(mp.cpu_count())

            # Step 2: `pool.apply` the `howmany_within_range()`
            _ = [
                pool.apply(self.single_trial_case3, args=(experiment_id, n, save_dir, False)) for experiment_id in setup
            ]

            # Step 3: Don't forget to close
            pool.close()
