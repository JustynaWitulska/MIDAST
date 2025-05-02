import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

import multiprocessing as mp
import os
import sys
from typing import Dict, List

sys.path.append("..")

from utils.stblrnd import sub_gaussian_vect_with_corr_change_case3, sub_gaussian_vect_with_corr_change_v2


class SubGaussianTrajectory:
    def __init__(self) -> None:
        """
        Initialize the SubGaussianTrajectory class with an empty setup dictionary.
        """
        self.setup: Dict[int, Dict[str, float]] = {}

    def set_setup(self, setup: Dict[int, Dict[str, float]]) -> None:
        """
        Set the experimental setup.

        Args:
            setup (Dict[int, Dict[str, float]]): A dictionary where keys are experiment IDs and values are dictionaries
                                                 containing parameters for the experiments.
        """
        self.setup = setup

    def single_trial(
        self,
        experiment_id: int,
        n: int,
        n_star: int,
        save_dir: str,
        save_fig: bool = False,
    ) -> pd.DataFrame:
        """
        Perform a single trial of sub-Gaussian vector simulation with correlation change.

        Args:
            experiment_id (int): ID of the experiment to run.
            n (int): Total number of samples.
            n_star (int): Index at which the correlation changes.
            save_dir (str): Directory to save the results.
            save_fig (bool, optional): Whether to save the figures. Defaults to False.

        Returns:
            pd.DataFrame: The resulting DataFrame without 'rho' and 'color' columns.
        """
        setup = self.setup.get(experiment_id)
        if setup:
            alpha = setup["alpha1"]
            alpha2 = setup["alpha2"]
            rho_before = setup["rho_before"]
            rho_after = setup["rho_after"]

            vect = sub_gaussian_vect_with_corr_change_v2(
                alpha=alpha,
                d=2,
                n=n,
                n_star=n_star,
                rho_before=rho_before,
                rho_after=rho_after,
                alpha2=alpha2,
            )

            df = pd.DataFrame(vect).rename(columns={0: "x", 1: "y"})
            df["rho"] = [rho_before] * n_star + [rho_after] * (df.shape[0] - n_star)
            df["color"] = df["rho"].apply(lambda x: "red" if x == rho_before else "blue")
            name = f"Simulation_n_{n}_alpha_{alpha}_alpha2_{alpha2}_rho1_{rho_before}_rho2_{rho_after}"

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

        return df.drop(columns=["rho", "color"])

    def single_trial_case3(self, experiment_id: int, n: int, save_dir: str, save_fig: bool = False) -> None:
        """
        Perform a single trial of sub-Gaussian vector simulation with three correlation changes.

        Args:
            experiment_id (int): ID of the experiment to run.
            n (int): Total number of samples.
            save_dir (str): Directory to save the results.
            save_fig (bool, optional): Whether to save the figures. Defaults to False.
        """
        setup = self.setup.get(experiment_id)
        if setup:
            alpha = setup["alpha1"]
            alpha2 = setup["alpha2"]
            alpha3 = setup["alpha3"]
            rho1 = setup["rho1"]
            rho2 = setup["rho2"]
            rho3 = setup["rho3"]
            n1 = int(0.33 * n)
            n2 = int(0.66 * n)
            n3 = n - n1 - n2

            vect = sub_gaussian_vect_with_corr_change_case3(
                alpha=alpha,
                d=2,
                n1=n1,
                n2=n2,
                n3=n3,
                rho1=rho1,
                rho2=rho2,
                rho3=rho3,
                alpha2=alpha2,
                alpha3=alpha3,
            )

            df = pd.DataFrame(vect).rename(columns={0: "x", 1: "y"})
            df["rho"] = [rho1] * n1 + [rho2] * n2 + [rho3] * n3
            df["color"] = df["rho"].replace(rho1, "green").replace(rho2, "red").replace(rho3, "blue").values
            name = f"Simulation_n_{n}_alpha_{alpha}_alpha2_{alpha2}_alpha3_{alpha3}_rho1_{rho1}_rho2_{rho2}_rho3_{rho3}"

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

    def make_simulations(self, setup: Dict[int, Dict[str, float]], n_list: List[int], case: int = 1) -> None:
        """
        Generate simulations for a given setup and list of sample sizes.

        Args:
            setup (Dict[int, Dict[str, float]]): Experimental setup dictionary.
            n_list (List[int], optional): List of sample sizes. Defaults to [200, 1000, 2000].
            case (int, optional): Case number for the simulation. Defaults to 1.
        """
        self.setup = setup
        for n in n_list:
            for coeff in [0.5]:  # 0.3
                n_star = int(coeff * n)
                save_dir = f"./case_1/Simulations_n_{n}_change_{n_star}"

                # Step 1: Init multiprocessing.Pool()
                pool = mp.Pool(mp.cpu_count())

                # Step 2: `pool.apply` the `howmany_within_range()`
                _ = [
                    pool.apply(
                        self.single_trial,
                        args=(experiment_id, n, n_star, save_dir, False),
                    )
                    for experiment_id in setup
                ]

                # Step 3: Don't forget to close
                pool.close()

    def make_simulations_case3(self, setup: Dict[int, Dict[str, float]], n_list: List[int]) -> None:
        """
        Generate simulations for case 3 with three correlation changes.

        Args:
            setup (Dict[int, Dict[str, float]]): Experimental setup dictionary.
            n_list (List[int], optional): List of sample sizes. Defaults to [300, 1000, 2000].
        """
        self.setup = setup
        for n in n_list:
            n1 = int(0.33 * n)
            n2 = int(0.66 * n)
            save_dir = f"./case_3/Simulations_n_{n}_{n1}_{n2}"

            # Step 1: Init multiprocessing.Pool()
            pool = mp.Pool(mp.cpu_count())

            # Step 2: `pool.apply` the `howmany_within_range()`
            _ = [
                pool.apply(self.single_trial_case3, args=(experiment_id, n, save_dir, False)) for experiment_id in setup
            ]

            # Step 3: Don't forget to close
            pool.close()
