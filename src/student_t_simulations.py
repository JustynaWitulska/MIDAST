import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

import multiprocessing as mp
import os
from typing import Dict, List, Optional, Union

from utils.stblrnd import t_student_vect_with_corr_change, t_student_vect_with_corr_change_case3


class StudentTrajectory:
    def __init__(self) -> None:
        """
        Initialize the StudentTrajectory class with an empty setup dictionary.
        """
        self.setup: Dict[int, Dict[str, Union[float, int]]] = {}

    def set_setup(self, setup: Dict[int, Dict[str, Union[float, int]]]) -> None:
        """
        Set the experimental setup.

        Args:
            setup (dict): A dictionary where keys are experiment IDs and values are dictionaries
                          containing parameters like degrees of freedom and correlation coefficients.
        """
        self.setup = setup

    def single_trial(
        self,
        experiment_id: int,
        n: int,
        n_star: int,
        save_dir: Optional[str] = None,
        save_fig: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Perform a single trial simulation for a two-phase t-student distribution.

        Args:
            experiment_id (int): ID of the experiment setup to use.
            n (int): Total number of samples.
            n_star (int): Number of samples in the first phase.
            save_dir (str, optional): Directory to save the results. Defaults to None.
            save_fig (bool, optional): Whether to save the generated figures. Defaults to False.

        Returns:
            pd.DataFrame or None: DataFrame containing the simulation results, or None if setup is not found.
        """
        setup = self.setup.get(experiment_id)
        if setup:
            dof1 = setup["dof1"]
            dof2 = setup["dof2"]
            rho_before = setup["rho_before"]
            rho_after = setup["rho_after"]

            vect = t_student_vect_with_corr_change(
                rho1=rho_before,
                rho2=rho_after,
                dof1=dof1,
                dof2=dof2,
                n=n,
                n_star=n_star,
            )

            df = pd.DataFrame(vect).rename(columns={0: "x", 1: "y"})
            df["rho"] = [rho_before] * n_star + [rho_after] * (df.shape[0] - n_star)
            df["dof"] = [dof1] * n_star + [dof2] * (df.shape[0] - n_star)
            df["color"] = df["rho"].apply(lambda x: "red" if x == rho_before else "blue")

            name = f"Simulation_n_{n}_dof1_{dof1}_dof2_{dof2}_rho1_{rho_before}_rho2_{rho_after}"

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

            if save_dir:
                if not os.path.exists(f"./{save_dir}"):
                    os.makedirs(f"./{save_dir}")

                df.to_csv(f"{save_dir}/{name}.csv")
                return None
            else:
                return df
        else:
            return None

    def single_trial_case3(self, experiment_id: int, n: int, save_dir: str, save_fig: bool = False) -> None:
        """
        Perform a single trial simulation for a three-phase t-student distribution.

        Args:
            experiment_id (int): ID of the experiment setup to use.
            n (int): Total number of samples.
            save_dir (str): Directory to save the results.
            save_fig (bool, optional): Whether to save the generated figures. Defaults to False.
        """
        setup = self.setup.get(experiment_id)
        if setup:
            dof1 = setup["dof1"]
            dof2 = setup["dof2"]
            dof3 = setup["dof3"]
            rho1 = setup["rho1"]
            rho2 = setup["rho2"]
            rho3 = setup["rho3"]
            n1 = int(0.33 * n)
            n2 = int(0.66 * n)
            n3 = n - n1 - n2

            vect = t_student_vect_with_corr_change_case3(
                rho1=rho1,
                rho2=rho2,
                rho3=rho3,
                dof1=dof1,
                dof2=dof2,
                dof3=dof3,
                n1=n1,
                n2=n2,
                n3=n3,
            )

            df = pd.DataFrame(vect).rename(columns={0: "x", 1: "y"})
            df["rho"] = [rho1] * n1 + [rho2] * n2 + [rho3] * n3
            df["color"] = df["rho"].replace(rho1, "green").replace(rho2, "red").replace(rho3, "blue").values
            name = f"Simulation_n_{n}_dof1_{dof1}_dof2_{dof2}_dof3_{dof3}_rho1_{rho1}_rho2_{rho2}_rho3_{rho3}"

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

    def make_simulations(
        self, setup: Dict[int, Dict[str, Union[float, int]]], n_list: List[int], case: int = 1
    ) -> None:
        """
        Generate simulations for a two-phase t-student distribution.

        Args:
            setup (dict): Experimental setup dictionary.
            n_list (list): List of sample sizes to simulate.
            case (int): Case identifier (currently unused).
        """
        self.set_setup(setup)
        for n in n_list:
            for coeff in [0.3, 0.5]:
                n_star = int(coeff * n)
                save_dir = f"./distr_t_student_one_cp/Simulations_n_{n}_change_{n_star}"
                pool = mp.Pool(mp.cpu_count())
                _ = [
                    pool.apply(
                        self.single_trial,
                        args=(experiment_id, n, n_star, save_dir, False),
                    )
                    for experiment_id in setup
                ]
                pool.close()

    def make_simulations_case3(self, setup: Dict[int, Dict[str, Union[float, int]]], n_list: List[int]) -> None:
        """
        Generate simulations for a three-phase t-student distribution.

        Args:
            setup (dict): Experimental setup dictionary.
            n_list (list): List of sample sizes to simulate.
        """
        self.setup = setup
        for n in n_list:
            n1 = int(0.33 * n)
            n2 = int(0.66 * n)
            save_dir = f"./distr_t_student_two_cp/Simulations_n_{n}_{n1}_{n2}"
            pool = mp.Pool(mp.cpu_count())
            _ = [
                pool.apply(self.single_trial_case3, args=(experiment_id, n, save_dir, False)) for experiment_id in setup
            ]
            pool.close()
