"""
Simulation of E. coli drift or entropy production and mutual information.

References:
    - Micali: Drift and Behavior of E. coli Cells
              Micali et al., Biophysical Journal, 2017
              https://www.sciencedirect.com/science/article/pii/S0006349517310755
    - Taylor: Information and fitness
              Taylor et al., arXiv, 2007
              https://arxiv.org/abs/0712.4382
    - Clausznitzer: Chemotactic Response and Adaptation Dynamics in Escherichia coli
                    Clausznitzer et al., PLOS Computational Biology, 2010
                    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000784
"""
from decimal import Decimal, ROUND_HALF_UP
from functools import partial
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import List, Literal, Set, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import trapezoid as integrate
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    outputs: Set[Literal['drift', 'entropy']] = {'drift', 'entropy'}
    """The outputs to optimize for when minimizing entropy. Drift is maximized while entropy is minimized."""
    num_iters: int = 200
    """Maximum number of iterations of the algorithm."""
    lambda_min: float = 0.0
    """Minimum value of Lagrangian lambda for drift in log space (i.e., min lambda = 10^{lambda_min})."""
    lambda_max: float = 3.0
    """Maximum value of Lagrangian lambda for drift in log space (i.e., min lambda = 10^{lambda_max})."""
    lambda_num: int = 9
    """Number of lambda values between lambda_min and lambda_max."""
    mu_min: float = -2.0
    """Minimum value of Lagrangian mu for entropy in log space (i.e., min mu = 10^{mu_min})."""
    mu_max: float = 2.0
    """Maximum value of Lagrangian mu for entropy in log space (i.e., min mu = 10^{mu_max})."""
    mu_num: int = 9
    """Number of mu values between mu_min and mu_max."""
    linearize_params: int = 0
    """The number of lambda/mu values to space linearly (rather than logarithmically) in the middle of the range."""
    ligand_gradient: float = 0.1
    """The relative gradient of the ligand concentration."""
    verbosity: Literal[0, 1, 2] = 1
    """Verbosity level. Higher means more verbose."""
    save_dir: Path = None
    """Directory where plots and arguments will be saved (if None, displayed instead)."""

    @staticmethod
    def linearize_array_middle(array: np.ndarray, num: int) -> np.ndarray:
        """
        Linearizes the middle portion of the array.

        Example:     [1e-4, 1e-3, 1e-2, 1e-1,    1,       1e1,      1e2, 1e3, 1e4]
                 ==> [1e-4, 1e-3, 1e-2, 2.5e+01, 5.0e+01, 7.50e+01, 1e2, 1e3, 1e4]

        :param array: An array of values.
        :param num: The number of values in the middle of the array to linearize.
        :return: An array of values with the middle portion of the values linearized.
        """
        # Copy params array
        array = array.copy()

        # Determine indices for middle portion
        middle = (len(array) - 1) / 2
        diff = num // 2
        start_index = int(Decimal(middle - diff).to_integral_value(rounding=ROUND_HALF_UP))
        end_index = int(Decimal(middle + diff).to_integral_value(rounding=ROUND_HALF_UP))

        if end_index - start_index == num:
            end_index -= 1

        # Replace middle portion with linear space
        array[start_index:end_index + 1] = np.linspace(array[start_index], array[end_index], num)

        return array

    @property
    def lams(self) -> np.ndarray:
        """
        Gets the range of values for the Lagrangian lambda for drift.

        :return: A numpy array with the range of lambda values (or just 0 if drift is not an output).
        """
        if 'drift' in self.outputs:
            lams = np.logspace(self.lambda_min, self.lambda_max, self.lambda_num)

            if self.linearize_params > 2:
                lams = self.linearize_array_middle(array=lams, num=self.linearize_params)
        else:
            lams = np.zeros(1)

        return lams

    @property
    def mus(self) -> np.ndarray:
        """
        Gets the range of values for the Lagrangian mu for entropy.

        :return: A numpy array with the range of mu values (or just 0 if entropy is not an output).
        """
        if 'entropy' in self.outputs:
            mus = np.logspace(self.mu_min, self.mu_max, self.mu_num)

            if self.linearize_params > 2:
                mus = self.linearize_array_middle(array=mus, num=self.linearize_params)
        else:
            mus = np.zeros(1)

        return mus

    @property
    def lagrangian_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets grids of Lagrangian lambda and mu values.

        :return: A tuple containing:
                   - lam_grid (np.ndarray): A numpy array of Lagrangian lambda values differing across the rows
                   - mu_grid (np.ndarray): A numpy array of Lagrangian mu values differing across the columns
        """
        mu_grid, lam_grid = np.meshgrid(self.mus, self.lams)

        return lam_grid, mu_grid

    def process_args(self) -> None:
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)


# Constants
CMAP = plt.get_cmap('viridis')
LEVELS = 64
EPS = np.spacing(1)
DPI = 300
METHYLATION_MIN = 0.0
METHYLATION_MAX = 8.0
DRIFT_UNITS = r'$\mu$m/s'
ENTROPY_UNITS = r'$\times 10^{-20}$ J $\cdot$ K$^{-1}$'
INFORMATION_UNITS = 'bits'
LIGAND_UNITS = 'mM'
FIGSIZE = np.array([6.4, 4.8])


def set_plot_parameters() -> None:
    """Sets matplotlib plot parameters."""
    # Plotting constants
    plt.rc('axes', titlesize=24)  # fontsize of the axes title
    plt.rc('axes', labelsize=18)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    plt.rc('legend', fontsize=14)  # legend fontsize


def set_up_methylation_levels_and_ligand_concentrations(m_min: float = METHYLATION_MIN,
                                                        m_max: float = METHYLATION_MAX,
                                                        m_num: int = 1000,
                                                        log_c_min: float = -3,
                                                        log_c_max: float = 3,
                                                        c_num: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sets up methylation levels and ligand concentrations.


    :param m_min: Minimum methylation level.
    :param m_max: Maximum methylation level.
    :param m_num: Number of methylation levels.
    :param log_c_min: Log base 10 of the minimum ligand concentration.
    :param log_c_max: Log base 10 of the maximum ligand concentration.
    :param c_num: Number of ligand concentrations.
    :return: A tuple containing:
               - m (np.ndarray): a matrix of methylation levels (differing across the rows)
               - c (np.ndarray): a matrix of ligand concentrations (differing across the columns)
    """
    mi = np.linspace(m_min, m_max, m_num)  # Methylation levels
    ci = np.logspace(log_c_min, log_c_max, c_num)  # Ligand concentrations (log space)

    c, m = np.meshgrid(ci, mi)  # Mesh grid of ligand concentrations and methylation levels

    return m, c


def compute_drift_and_entropy_production(c: np.ndarray,
                                         m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the drift and entropy production.

    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :return: A tuple consisting of a matrix of drift values and a matrix of entropy production values.
    """
    # Parameters (Micali table S1)
    N = 5  # Cooperative receptor number (paper uses 13, range[5; 13])
    va = 1 / 3  # Fraction of Tar receptors (paper uses 1 / 3)
    vs = 2 / 3  # Fraction of Tsr receptors (paper uses 2 / 3)
    Kaon = 0.5  # Active receptors dissociation constant Tar (mM, paper uses 1.0)
    Kson = 1e6  # Active receptors dissociation constant Tsr (mM, paper uses 1E6)
    Kaoff = 0.02  # Inactive receptors dissociation constant Tar (mM, paper uses 0.03)
    Ksoff = 100  # Inactive receptors dissociation constant Tsr (mM, paper uses 100)
    YT = 9.7  # Total concentration of CheY (uM, paper uses 7.9, range [6; 9.7])
    k = 1  # Approximation of the susceptibility function.

    # Phosphorylation rate parameters (Micali equation S28)
    nu = 0.01  # Proportionality constant between forward and backward rates (paper says << 1)

    kyp = 100  # Rate constant k_y^+ for reaction CheY + ATP --> CheY_p + ADP (uM^{-1} * s^{-1})
    kym = nu * kyp  # Rate constant k_y^- for reaction CheY_p + ADP --> CheY + ATP (uM^{-1} * s^{-1})

    kzp = 30  # Rate constant k_z^+ for reaction CheY_p + ADP --> CheY + ADP + Pi (s^{-1})
    kzm = nu * kzp  # Rate constant k_z^- for reaction CheY + ADP + Pi --> CheY_p + ADP (s^{-1})

    # Methylation rate parameters (Micali equation S29)
    kRp = 0.0069  # Rate constant k_R^+ for reaction [m]_0 + SAM --> [m + 1]_0 + SAH (s^{-1})
    kRm = nu * kRp  # Rate constant k_R^- for reaction [m + 1]_0 + SAH --> [m]_0 + SAM (s^{-1})
    kBp = 0.12  # Rate constant k_B^+ for reaction [m + 1]_1 + H2O --> [m]_1 + CH3OH (s^{-1})
    kBm = nu * kBp  # Rate constant k_B^+ for reaction [m]_1 + CH3OH --> [m + 1]_1 + H2O (s^{-1})
    mT = 1e4  # Total number of methylation sites

    # Physical constant
    kb = 1.380649 / 1000  # Boltzmann constant (times 10^{-20} J * K^{-1})

    # Ligand (local) gradient steepness (mm^{-1})
    rel_grad = 0.75

    # Methylation free energy (Monod-Wyman-Changeux (MWC) model, Clausznitzer equation 5)
    mE = 1 - 0.5 * m

    # Free energy difference of on/off states of receptor complex (MWC model, Clausznitzer equation 5)
    dF = N * (  # cooperative receptor number
            mE +  # methylation energy
            va * np.log((1 + c / Kaoff) / (1 + c / Kaon)) +  # Tar receptor
            vs * np.log((1 + c / Ksoff) / (1 + c / Kson))  # Tsr receptor
    )

    # Receptor activity (MWC model, Clausznitzer equation 5)
    A = 1 / (1 + np.exp(dF))

    # Derivative of free energy with respect to ligand concentration (Micali equation 1 and S20)
    dFdc = N * (  # cooperative receptor number
            va * (c / (c + Kaoff) - c / (c + Kaon)) +  # Tar receptor
            vs * (c / (c + Ksoff) - c / (c + Kson))  # Tsr receptor
    )

    # Drift velocity in um/s (Micali equation 1)
    drift = k * A * (1 - A) * dFdc * rel_grad

    # Concentration of phosphorylated CheY (CheY_p)
    # from CheY + ATP --> CheY_p + ADP (kyp rate constant) and CheY + ADP + Pi --> CheY_p + ADP (kzm rate constant)
    # (Micali derived from setting dYp/dt = 0 for the equation between equations S27 and S28)
    Yp = (kyp * A + kzm) * YT / ((kyp + kym) * A + kzp + kzm)

    # Entropy production of the phosphorylation dynamics (Micali equation S28)
    dSdty = kb * ((kyp * (YT - Yp) - kym * Yp) * A * np.log((kyp * (YT - Yp)) / (kym * Yp)) +
                  (kzp * Yp - kzm * (YT - Yp)) * np.log((kzp * Yp) / (kzm * (YT - Yp))))

    # Entropy production of the methylation dynamics (Micali equation S29)
    dSdtm = kb * ((kRp - kRm) * (1 - A) * mT * np.log(kRp / kRm) +
                  (kBp - kBm) * A**3 * mT * np.log(kBp / kBm))

    # Entropy production of the phosphorylation and methylation dynamics in J * K^{-1} (times 10^{-20})
    entropy_production = (dSdty + dSdtm)  # Micali equation S29

    return drift, entropy_production


def plot_output(output: np.ndarray,
                title: str,
                c: np.ndarray,
                m: np.ndarray,
                plot_max: bool = False,
                save_path: Path = None) -> None:
    """
    Plots the output given the ligand concentration and methylation level.

    :param output: A matrix containing the output of interest for different methylation levels (rows)
                   and ligand concentrations (columns).
    :param title: The name of the type of output.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param plot_max: Whether to plot the maximum y value for each x value.
    :param save_path: Path where the plot will be saved (if None, displayed instead).
    """
    log_c = np.log10(c)

    fig, ax = plt.subplots(figsize=FIGSIZE * 1.2)
    im = ax.contourf(log_c, m, output, levels=LEVELS, cmap=CMAP, zorder=-20)
    ax.set_rasterization_zorder(-10)
    fig.colorbar(im, ax=ax)

    if plot_max:
        maxi = np.argmax(output, axis=0)  # Index in each column corresponding to maximum
        ax.plot(log_c[0], m[maxi, 0], color='red', label='max')
        ax.legend(loc='upper left')

    ax.set_title(title)
    ax.set_xlabel(r'Ligand concentration $\log_{10}(c)$ ' + f'({LIGAND_UNITS})')
    ax.set_ylabel('Methylation level $m$')

    if save_path is not None:
        plt.savefig(save_path, dpi=DPI)
    else:
        plt.show()

    plt.close()


def set_up_ligand_concentration_distribution(c: np.ndarray, ligand_gradient: float) -> np.ndarray:
    """
    Sets up the marginal distribution P(c) over ligand concentrations.

    :param c: A matrix of ligand concentrations (differing across the columns).
    :param ligand_gradient: The relative gradient of the ligand concentration.
    :return: A matrix containing the marginal distribution P(c) over ligand concentrations.
    """
    pc = np.exp(-ligand_gradient * c)
    Pc = pc / integrate(pc, c, axis=1)

    return Pc


def compute_Pm(Pmc: np.ndarray,
               Pc: np.ndarray,
               c: np.ndarray) -> np.ndarray:
    """
    Given the conditional distribution P(m | c), computes the marginal distribution P(m).

    :param Pmc: The conditional distribution P(m | c) over methylation levels given ligand concentrations.
    :param Pc: The marginal distribution P(c) over ligand concentrations.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :return: The marginal distribution P(m) over methylation levels,
             which is a matrix of size (num_methylation_levels, num_ligand_concentrations).
    """
    return np.broadcast_to(integrate(Pmc * Pc, c, axis=1)[:, np.newaxis], Pmc.shape)


def compute_Pmc(Pm: np.ndarray,
                m: np.ndarray,
                exp_output: np.ndarray) -> np.ndarray:
    """
    Given the marginal distribution P(m), computes the conditional distribution P(m | c).

    :param Pm: The marginal distribution P(m) over methylation levels.
    :param m: A matrix of methylation levels (differing across the rows).
    :param exp_output: A matrix containing the exponential output.
    :return: The conditional distribution P(m | c) over methylation levels given ligand concentrations,
             which is a matrix of size (num_methylation_levels, num_ligand_concentrations).
    """
    pmc = Pm * exp_output
    Pmc = pmc / integrate(pmc, m, axis=0)

    return Pmc


def plot_values_across_iterations(infos: List[float],
                                  avg_drifts: List[float],
                                  avg_entropies: List[float],
                                  objectives: List[float],
                                  lam: float,
                                  mu: float,
                                  save_path: Path = None) -> None:
    """
    Plots the mutual information, average drift, average entropy, and objective function across iterations.

    :param infos: A list of mutual information values across iterations.
    :param avg_drifts: A list of average drift values across iterations.
    :param avg_entropies: A list of average entropy values across iterations.
    :param objectives: A list of objective function values across iterations.
    :param lam: The Lagrangian lambda for drift.
    :param mu: The Lagrangian mu for entropy.
    :param save_path: Path where the plot will be saved (if None, displayed instead).
    """
    s = 3
    fig, axes = plt.subplots(4, 1, sharex=True)

    axes[0].scatter(np.arange(len(infos)), infos,
                    color='red', label=f'Information ({INFORMATION_UNITS})', s=s)
    axes[1].scatter(np.arange(len(avg_drifts)), avg_drifts,
                    color='blue', label=f'Drift ({DRIFT_UNITS})', s=s)
    axes[2].scatter(np.arange(len(avg_entropies)), avg_entropies,
                    color='purple', label=f'Entropy ({ENTROPY_UNITS})', s=s)
    axes[3].scatter(np.arange(len(objectives)), objectives,
                    color='green', label=f'Objective', s=s)

    for ax in axes:
        ax.legend()

    plt.xlabel('Iteration')
    axes[0].set_title(rf'Values across iterations for $\lambda = {lam:.2e}, \mu = {mu:.2e}$')

    if save_path is not None:
        plt.savefig(save_path, dpi=DPI)
    else:
        plt.show()

    plt.close()


def compute_mutual_information(Pmc: np.ndarray,
                               Pc: np.ndarray,
                               Pm: np.ndarray,
                               c: np.ndarray,
                               m: np.ndarray) -> float:
    """
    Computes the mutual information of the given conditional and marginal distributions.

    :param Pmc: The conditional distribution P(m | c) over methylation levels given ligand concentrations.
    :param Pc: The marginal distribution P(c) over ligand concentrations.
    :param Pm: The marginal distribution P(m) over methylation levels.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :return: The mutual information of the given conditional and marginal distributions.
    """
    return integrate(Pc * integrate(Pmc * np.log2(EPS + Pmc / (EPS + Pm)), m, axis=0), c, axis=1)[0]


def compute_average_output(output: np.ndarray,
                           Pmc: np.ndarray,
                           Pc: np.ndarray,
                           c: np.ndarray,
                           m: np.ndarray) -> float:
    """
    Computes the average output given the conditional and marginal distributions.

    :param output: A matrix containing the output of interest
                   for different methylation levels (rows) and ligand concentrations (columns).
    :param Pmc: The conditional distribution P(m | c) over methylation levels given ligand concentrations.
    :param Pc: The marginal distribution P(c) over ligand concentrations.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :return: The average output given the conditional and marginal distributions.
    """
    return integrate(Pc * integrate(Pmc * output, m, axis=0), c, axis=1)[0]


def compute_standard_deviation_output(output: np.ndarray,
                                      Pmc: np.ndarray,
                                      Pc: np.ndarray,
                                      c: np.ndarray,
                                      m: np.ndarray,
                                      avg_output: float = None) -> float:
    """
    Computes the standard deviation of the output given the conditional and marginal distributions.

    :param output: A matrix containing the output of interest
                   for different methylation levels (rows) and ligand concentrations (columns).
    :param Pmc: The conditional distribution P(m | c) over methylation levels given ligand concentrations.
    :param Pc: The marginal distribution P(c) over ligand concentrations.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param avg_output: The average output. If None, the average output is computed.
    :return: The standard deviation of the output given the conditional and marginal distributions.
    """
    if avg_output is None:
        avg_output = compute_average_output(
            output=output,
            Pmc=Pmc,
            Pc=Pc,
            c=c,
            m=m
        )

    avg_squared_output = compute_average_output(
        output=output ** 2,
        Pmc=Pmc,
        Pc=Pc,
        c=c,
        m=m
    )

    std_output = np.sqrt(avg_squared_output - avg_output ** 2)

    return std_output


def determine_information_and_output(drift: np.ndarray,
                                     entropy: np.ndarray,
                                     Pc: np.ndarray,
                                     c: np.ndarray,
                                     m: np.ndarray,
                                     lam: float,
                                     mu: float,
                                     num_iters: int,
                                     verbosity: int,
                                     save_dir: Path = None) -> Tuple[float, float, float, float, float,
                                                                     np.ndarray, np.ndarray]:
    """
    Iterates an algorithm to determine the minimum mutual information and maximum mean fitness.

    :param drift: A numpy array containing the drift
                   for different methylation levels (rows) and ligand concentrations (columns).
    :param entropy: A numpy array containing the entropy
                   for different methylation levels (rows) and ligand concentrations (columns).
    :param Pc: The marginal distribution P(c) over ligand concentrations.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param lam: The Lagrangian lambda for drift.
    :param mu: The Lagrangian mu for entropy.
    :param num_iters: Maximum number of iterations of the algorithm.
    :param verbosity: Verbosity level. Higher means more verbose.
    :param save_dir: Directory where the plot will be saved (if None, displayed instead).
    :return: A tuple containing:
               - info (float): the (minimum) mutual information
               - avg_drift (float): the (maximum) average drift
               - std_drift (float): the standard deviation of the drift
               - avg_entropy (float): the (minimum) average entropy
               - std_entropy (float): the standard deviation of the entropy
               - Pmc (np.ndarray): the conditional distributions P(m | c)
               - Pm (np.ndarray): the marginal distributions P(m)
    """
    # Keep track of values across iterations (if verbosity >= 2)
    infos, avg_drifts, avg_entropies, objectives = [], [], [], []

    # Initial guess for marginal distribution P(m) over methylation levels
    Pm = np.ones(Pc.shape)

    # Normalize P(m)
    Pm = Pm / integrate(Pm, m, axis=0)

    # Precompute 2^(lam * drift - mu * entropy)
    exp_output = np.power(2, lam * drift - mu * entropy)

    # Initial guess for conditional distribution P(m | c) over methylation levels given ligand concentrations
    Pmc = compute_Pmc(Pm=Pm, m=m, exp_output=exp_output)

    # Iterate algorithm
    for _ in range(num_iters):
        # Compute new P(m)
        Pm = compute_Pm(Pmc=Pmc, Pc=Pc, c=c)

        # Compute new P(m | c)
        Pmc = compute_Pmc(Pm=Pm, m=m, exp_output=exp_output)

        # Keep track of I, out, and objective function across iterations
        if verbosity >= 2:
            info = compute_mutual_information(Pmc=Pmc, Pc=Pc, Pm=Pm, c=c, m=m)
            avg_drift = compute_average_output(output=drift, Pmc=Pmc, Pc=Pc, c=c, m=m)
            avg_entropy = compute_average_output(output=entropy, Pmc=Pmc, Pc=Pc, c=c, m=m)
            objective = info - lam * avg_drift + mu * avg_entropy

            infos.append(info)
            avg_drifts.append(avg_drift)
            avg_entropies.append(avg_entropy)
            objectives.append(objective)

    # Plot I, out, and objective function across iterations
    if verbosity >= 2:
        if save_dir is not None:
            save_dir = save_dir / 'convergence'
            save_dir.mkdir(parents=True, exist_ok=True)

        plot_values_across_iterations(
            infos=infos,
            avg_drifts=avg_drifts,
            avg_entropies=avg_entropies,
            objectives=objectives,
            lam=lam,
            mu=mu,
            save_path=save_dir / f'convergence_lambda={lam:.2e}_mu={mu:.2e}.pdf' if save_dir is not None else None
        )

    # Compute the minimum mutual information (Taylor equation 1)
    info = compute_mutual_information(Pmc=Pmc, Pc=Pc, Pm=Pm, c=c, m=m)

    # Compute maximum mean output and standard deviation (Taylor equation 4)
    avg_drift = compute_average_output(output=drift, Pmc=Pmc, Pc=Pc, c=c, m=m)
    std_drift = compute_standard_deviation_output(output=drift, Pmc=Pmc, Pc=Pc, c=c, m=m, avg_output=avg_drift)
    avg_entropy = compute_average_output(output=entropy, Pmc=Pmc, Pc=Pc, c=c, m=m)
    std_entropy = compute_standard_deviation_output(output=entropy, Pmc=Pmc, Pc=Pc, c=c, m=m, avg_output=avg_entropy)

    return info, avg_drift, std_drift, avg_entropy, std_entropy, Pmc, Pm


def determine_information_and_output_multiprocessing(i_j: Tuple[int, int],
                                                     lam_grid: np.ndarray,
                                                     mu_grid: np.ndarray,
                                                     **kwargs) -> Tuple[Tuple[int, int],
                                                                        Tuple[float, float, float, float, float,
                                                                              np.ndarray, np.ndarray]]:
    """Multiprocessing-friendly version of determine_information_and_output."""
    i, j = i_j
    return (i, j), determine_information_and_output(
        lam=lam_grid[i, j],
        mu=mu_grid[i, j],
        **kwargs
    )


def grid_search_information_and_output(drift: np.ndarray,
                                       entropy: np.ndarray,
                                       Pc: np.ndarray,
                                       c: np.ndarray,
                                       m: np.ndarray,
                                       lam_grid: np.ndarray,
                                       mu_grid: np.ndarray,
                                       num_iters: int,
                                       verbosity: int,
                                       save_dir: Path = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                       np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterates an algorithm to determine the minimum mutual information and maximum mean fitness for different parameters.

    :param drift: A numpy array containing the drift
                   for different methylation levels (rows) and ligand concentrations (columns).
    :param entropy: A numpy array containing the entropy
                   for different methylation levels (rows) and ligand concentrations (columns).
    :param Pc: The marginal distribution P(c) over ligand concentrations.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param lam_grid: A numpy array of Lagrangian mu values differing across the rows.
    :param mu_grid: A numpy array of Lagrangian mu values differing across the columns.
    :param num_iters: Maximum number of iterations of the algorithm.
    :param verbosity: Verbosity level. Higher means more verbose.
    :param save_dir: Directory where the plot will be saved (if None, displayed instead).
    :return: A tuple containing:
               - info_grid (np.ndarray): a matrix of (minimum) mutual information values for each lambda and mu
               - avg_drift_grid (np.ndarray): a matrix of (maximum) average drift values for each lambda and mu
               - std_drift_grid (np.ndarray): a matrix of drift standard deviation values for each lambda and mu
               - avg_entropy_grid (np.ndarray): a matrix of (minimum) average entropy values for each lambda and mu
               - std_entropy_grid (np.ndarray): a matrix of entropy standard deviation values for each lambda and mu
               - Pmc_grid (np.ndarray): a matrix of  conditional distributions P(m | c) for each lambda and mu
               - Pm_grid (np.ndarray): a matrix of marginal distributions P(m) for each lambda and mu
    """
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    determine_information_and_output_multiprocessing_fn = partial(
        determine_information_and_output_multiprocessing,
        drift=drift,
        entropy=entropy,
        Pc=Pc,
        c=c,
        m=m,
        lam_grid=lam_grid,
        mu_grid=mu_grid,
        num_iters=num_iters,
        verbosity=verbosity,
        save_dir=save_dir
    )

    # Get grid shapes
    lagrangian_grid_shape = lam_grid.shape
    input_grid_shape = c.shape

    # Set up grids to collect information and distributions
    info_grid = np.zeros(lagrangian_grid_shape)
    avg_drift_grid = np.zeros(lagrangian_grid_shape)
    std_drift_grid = np.zeros(lagrangian_grid_shape)
    avg_entropy_grid = np.zeros(lagrangian_grid_shape)
    std_entropy_grid = np.zeros(lagrangian_grid_shape)
    Pmc_grid = np.zeros((*lagrangian_grid_shape, *input_grid_shape))
    Pm_grid = np.zeros((*lagrangian_grid_shape, *input_grid_shape))

    with Pool() as p:
        for (i, j), results in tqdm(p.imap(determine_information_and_output_multiprocessing_fn,
                                           product(range(lam_grid.shape[0]), range(lam_grid.shape[1]))),
                                    total=lam_grid.size):
            info_grid[i, j], avg_drift_grid[i, j], std_drift_grid[i, j], avg_entropy_grid[i, j], \
            std_entropy_grid[i, j], Pmc_grid[i, j], Pm_grid[i, j] = results

    return info_grid, avg_drift_grid, std_drift_grid, avg_entropy_grid, std_entropy_grid, Pmc_grid, Pm_grid


def plot_information_and_output(infos: List[float],
                                avg_outs: List[float],
                                output_type: str,
                                units: str,
                                save_path: Path = None) -> None:
    """
    Plot the (maximum/minimum) mean output vs the (minimum) mutual information.

    :param infos: A list of (minimum) mutual information values.
    :param avg_outs: A list of (maximum/minimum) average output values.
    :param output_type: The name of the type of output.
    :param units: Units of the output type.
    :param save_path: Path where the plot will be saved (if None, displayed instead).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE * 1.2)

    ax.plot(infos, avg_outs, 'x')
    ax.set_title(f'{output_type} vs Mutual Information')
    ax.set_ylabel(f'{output_type} ({units})')
    ax.set_xlabel(f'Mutual Information ({INFORMATION_UNITS})')

    if save_path is not None:
        plt.savefig(save_path, dpi=DPI)
    else:
        plt.show()

    plt.close()


def plot_information_and_outputs_3d_on_axis(ax: Axes3D,
                                            info_grid: np.ndarray,
                                            avg_drift_grid: np.ndarray,
                                            avg_entropy_grid: np.ndarray) -> None:
    """
    Plot the (maximum) average drift vs the (minimum) average entropy vs the (minimum) mutual information on an axis.

    :param ax: A 3D axis object on which the plot will be created.
    :param info_grid: A matrix of mutual information values for different lambda and mu values.
    :param avg_drift_grid: A matrix of average drift values for different lambda and mu values.
    :param avg_entropy_grid: A matrix of average entropy values for different lambda and mu values.
    """
    # Numbers from low mu to high mu (connected by a line)
    # Blue to green from low lambda to high lambda
    for i, (info_row, avg_drift_row, avg_entropy_row) in enumerate(zip(info_grid, avg_drift_grid, avg_entropy_grid)):
        color = CMAP(i / len(info_row))
        ax.plot(info_row, avg_drift_row, avg_entropy_row, c=color)

        for j, (info, avg_drift, avg_entropy) in enumerate(zip(info_row, avg_drift_row, avg_entropy_row)):
            ax.text(info, avg_drift, avg_entropy, str(j + 1), color=color)

    ax.set_xlabel(f'Mutual Information ({INFORMATION_UNITS})', fontsize=12)
    ax.set_ylabel(f'Drift ({DRIFT_UNITS})', fontsize=12)
    ax.set_zlabel(f'Entropy ({ENTROPY_UNITS})', fontsize=12)


def plot_information_and_outputs_3d(info_grid: np.ndarray,
                                    avg_drift_grid: np.ndarray,
                                    avg_entropy_grid: np.ndarray,
                                    save_path: Path = None) -> None:
    """
    Plot the (maximum) average drift vs the (minimum) average entropy vs the (minimum) mutual information.

    :param info_grid: A matrix of mutual information values for different lambda and mu values.
    :param avg_drift_grid: A matrix of average drift values for different lambda and mu values.
    :param avg_entropy_grid: A matrix of average entropy values for different lambda and mu values.
    :param save_path: Path where the plots will be saved (if None, displayed instead).
    """
    if save_path is not None:
        fig = plt.figure(figsize=FIGSIZE * 3)
        num_rows = num_cols = 3
        num_images = num_rows * num_cols
        for index, angle in enumerate(range(0, 360, 360 // num_images)):
            ax = fig.add_subplot(num_rows, num_cols, index + 1, projection='3d')
            ax.view_init(azim=ax.azim + angle)
            plot_information_and_outputs_3d_on_axis(
                ax=ax,
                info_grid=info_grid,
                avg_drift_grid=avg_drift_grid,
                avg_entropy_grid=avg_entropy_grid
            )

        fig.suptitle('Drift vs Entropy vs Mutual Information', fontsize=50)
        plt.savefig(save_path, dpi=DPI)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plot_information_and_outputs_3d_on_axis(
            ax=ax,
            info_grid=info_grid,
            avg_drift_grid=avg_drift_grid,
            avg_entropy_grid=avg_entropy_grid
        )
        plt.title('Drift vs Entropy vs Mutual Information')
        plt.show()

    plt.close()


def avg_plus_minus_std_string(avg: float, std: float, precision: int = 2) -> str:
    """
    Creates a string of the form 'avg +/- std' using scientific notation with the precision provided.

    :param avg: The average value.
    :param std: The standard deviation value.
    :param precision: The floating point precision to use.
    :return: A string of the form 'avg +/- std' using scientific notation with the precision provided.
    """
    avg_power = int(np.floor(np.log10(avg)))
    std_power = int(np.floor(np.log10(std)))
    power = max(avg_power, std_power)
    base = 10 ** power
    sign = '+' if power >= 0 else '-'
    power_string = str(abs(power)).zfill(2)
    string = rf'{avg / base:.{precision}f}$\pm${std / base:.{precision}f}e{sign}{power_string}'

    return string


def plot_distributions_across_parameters(distributions: np.ndarray,
                                         c: np.ndarray,
                                         m: np.ndarray,
                                         parameters: np.ndarray,
                                         parameter_name: str,
                                         infos: List[float],
                                         avg_outs: List[float],
                                         std_outs: List[float],
                                         output_type: str,
                                         units: str,
                                         title: str,
                                         save_path: Path = None) -> None:
    """
    Plots the distributions over methylation levels and ligand concentrations across parameter values.

    :param distributions: An array of probability distributions over methylation levels
                          and ligand concentrations across parameter values.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param parameters: A list of parameter values for each distribution.
    :param parameter_name: The name of the parameter corresponding to the values in parameters.
    :param infos: A list of minimum mutual information values.
    :param avg_outs: A list of maximum mean output values.
    :param std_outs: A list of output standard deviation values.
    :param output_type: The name of the type of output.
    :param units: Units of the output type.
    :param title: The title of the plot.
    :param save_path: Path where the plot will be saved (if None, displayed instead).
    """
    log_c = np.log10(c)
    sqrt_size = int(np.ceil(np.sqrt(len(parameters))))  # Number of rows/columns in a square that can hold all the plots

    fig, axes = plt.subplots(nrows=sqrt_size, ncols=sqrt_size, figsize=sqrt_size * FIGSIZE * 1.2)
    axes = [axes] if sqrt_size == 1 else axes.flat

    for ax, distribution, parameter, info, avg_out, std_out in tqdm(zip(axes, distributions, parameters,
                                                                        infos, avg_outs, std_outs),
                                                                    total=len(parameters)):
        im = ax.contourf(log_c, m, distribution, levels=LEVELS, cmap=CMAP, zorder=-20)
        ax.set_rasterization_zorder(-10)

        fig.colorbar(im, ax=ax)
        ax.set_title(f'{parameter_name}$=${parameter:.2e}, '
                     f'I$=${info:.2e} {INFORMATION_UNITS},\n'
                     rf'{output_type}$=${avg_plus_minus_std_string(avg_out, std_out)} {units}',
                     fontsize=5 * sqrt_size)

    fig.suptitle(title, fontsize=20 * sqrt_size)
    fig.text(0.04, 0.5, 'Methylation level $m$',
             va='center', rotation='vertical', fontsize=18 * sqrt_size)  # y label
    fig.text(0.5, 0.04, r'Ligand concentration $\log_{10}(c)$ ' + f'({LIGAND_UNITS})',
             ha='center', fontsize=18 * sqrt_size)  # x label

    if save_path is not None:
        plt.savefig(save_path, dpi=DPI)
    else:
        plt.show()

    plt.close()


def plot_distributions_across_parameter_grid(distribution_grid: np.ndarray,
                                             c: np.ndarray,
                                             m: np.ndarray,
                                             lam_grid: np.ndarray,
                                             mu_grid: np.ndarray,
                                             title: str,
                                             info_grid: np.ndarray = None,
                                             avg_drift_grid: np.ndarray = None,
                                             std_drift_grid: np.ndarray = None,
                                             avg_entropy_grid: np.ndarray = None,
                                             std_entropy_grid: np.ndarray = None,
                                             plot_max: bool = False,
                                             save_path: Path = None) -> None:
    """
    Plots the distributions over methylation levels and ligand concentrations across parameter values.

    :param distribution_grid: A matrix of probability distributions over methylation levels
                              and ligand concentrations across parameter values.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param lam_grid: A numpy array of Lagrangian mu values differing across the rows.
    :param mu_grid: A numpy array of Lagrangian mu values differing across the columns.
    :param title: The title of the plot.
    :param info_grid: A matrix of mutual information values for different lambda and mu values.
    :param avg_drift_grid: A matrix of average drift values for different lambda and mu values.
    :param std_drift_grid: A matrix of drift standard deviation values for each lambda and mu.
    :param avg_entropy_grid: A matrix of average entropy values for different lambda and mu values.
    :param std_entropy_grid: A matrix of entropy standard deviation values for each lambda and mu.
    :param plot_max: Whether to plot the maximum y value for each x value.
    :param save_path: Path where the plot will be saved (if None, displayed instead).
    """
    log_c = np.log10(c)
    nrows, ncols = lam_grid.shape
    size = lam_grid.size
    sqrt_size = np.sqrt(size)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=sqrt_size * FIGSIZE * 1.2)

    # Ensure axes is a 2d array
    if size == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    # Plot distributions
    for i, j in tqdm(product(range(nrows), range(ncols)), total=size):
        ax = axes[i, j]
        im = ax.contourf(log_c, m, distribution_grid[i, j], levels=LEVELS, cmap=CMAP, zorder=-20)
        ax.set_rasterization_zorder(-10)

        fig.colorbar(im, ax=ax)
        ax.set_title((f'I$=${info_grid[i, j]:.2e} {INFORMATION_UNITS}, ' if info_grid is not None else '') +
                     (f'V$=${avg_plus_minus_std_string(avg_drift_grid[i, j], std_drift_grid[i, j])} {DRIFT_UNITS},\n'
                      if avg_drift_grid is not None and std_drift_grid is not None else '') +
                     (f'S$=${avg_plus_minus_std_string(avg_entropy_grid[i, j], std_entropy_grid[i, j])} {ENTROPY_UNITS}'
                      if avg_entropy_grid is not None and std_entropy_grid is not None else ''),
                     fontsize=15)

        if plot_max:
            maxi = np.argmax(distribution_grid[i, j], axis=0)  # Index in each column corresponding to maximum
            ax.plot(log_c[0], m[maxi, 0], color='red', label='max')
            ax.legend(loc='upper left')

        if i == nrows - 1:
            ax.set_xlabel(rf'$\mu=${mu_grid[i, j]:.2e}', fontsize=24 if sqrt_size <= 6 else 40)

        if j == 0:
            ax.set_ylabel(rf'$\lambda=${lam_grid[i, j]:.2e}', fontsize=24 if sqrt_size <= 6 else 40)

    fig.suptitle(title, fontsize=20 * sqrt_size)
    fig.text(0.04, 0.5, 'Methylation level $m$',
             va='center', rotation='vertical', fontsize=18 * sqrt_size)  # y label
    fig.text(0.5, 0.04, r'Ligand concentration $\log_{10}(c)$ ' + f'({LIGAND_UNITS})',
             ha='center', fontsize=18 * sqrt_size)  # x label

    if save_path is not None:
        plt.savefig(save_path, dpi=DPI)
    else:
        plt.show()

    plt.close()


def run_simulation(args: Args) -> None:
    """Runs the rate distortion simulation."""
    # Set plot parameters
    set_plot_parameters()

    # Save arguments
    if args.save_dir is not None:
        args.save(args.save_dir / 'args.json')

    # Get grid of Lagrangian lambda and mu values
    lam_grid, mu_grid = args.lagrangian_grid

    # Set up the methylation levels and ligand concentrations
    m, c = set_up_methylation_levels_and_ligand_concentrations()

    # Compute drift and entropy
    drift, entropy = compute_drift_and_entropy_production(c=c, m=m)

    # Plot outputs
    if args.verbosity >= 1:
        if args.outputs == {'drift'}:
            plot_output(
                output=drift,
                title=f'Drift ({DRIFT_UNITS})',
                c=c,
                m=m,
                plot_max=True,
                save_path=args.save_dir / 'drift.pdf' if args.save_dir is not None else None
            )

        elif args.outputs == {'entropy'}:
            plot_output(
                output=entropy,
                title=f'Entropy ({ENTROPY_UNITS})',
                c=c,
                m=m,
                plot_max=True,
                save_path=args.save_dir / 'entropy.pdf' if args.save_dir is not None else None
            )

        elif args.outputs == {'drift', 'entropy'}:
            # Plot lambda * drift - mu * entropy
            plot_distributions_across_parameter_grid(
                distribution_grid=lam_grid.reshape((*lam_grid.shape, 1, 1)) * drift.reshape((1, 1, *drift.shape)) -
                                  mu_grid.reshape((*mu_grid.shape, 1, 1)) * entropy.reshape((1, 1, *entropy.shape)),
                c=c,
                m=m,
                lam_grid=lam_grid,
                mu_grid=mu_grid,
                title=r'$\lambda \cdot$ drift $-\ \mu \cdot$ entropy',
                plot_max=True,
                save_path=args.save_dir / 'drift-entropy.pdf' if args.save_dir is not None else None
            )

    # Set up marginal distribution over ligand concentrations P(c)
    Pc = set_up_ligand_concentration_distribution(c=c, ligand_gradient=args.ligand_gradient)

    # Plot P(c)
    if args.verbosity >= 1:
        plot_output(
            output=Pc,
            title='$P(c)$',
            c=c,
            m=m,
            save_path=args.save_dir / 'pc.pdf' if args.save_dir is not None else None
        )

    # Determine minimum mutual information and maximum mean output for multiple parameter values
    info_grid, avg_drift_grid, std_drift_grid, avg_entropy_grid, std_entropy_grid, Pmc_grid, Pm_grid = \
        grid_search_information_and_output(
            drift=drift,
            entropy=entropy,
            Pc=Pc,
            c=c,
            m=m,
            lam_grid=lam_grid,
            mu_grid=mu_grid,
            num_iters=args.num_iters,
            verbosity=args.verbosity,
            save_dir=args.save_dir
        )

    # Plot average output(s) vs mutual information and plot distributions
    if args.outputs == {'drift', 'entropy'}:
        # Plot average drift vs average entropy vs mutual information
        plot_information_and_outputs_3d(
            info_grid=info_grid,
            avg_drift_grid=avg_drift_grid,
            avg_entropy_grid=avg_entropy_grid,
            save_path=args.save_dir / 'drift_vs_entropy_vs_information.pdf' if args.save_dir is not None else None
        )

        # Set up distribution plotting function across Lagrangian values
        plot_dist_fn = partial(
            plot_distributions_across_parameter_grid,
            c=c,
            m=m,
            lam_grid=lam_grid,
            mu_grid=mu_grid,
            info_grid=info_grid,
            avg_drift_grid=avg_drift_grid,
            std_drift_grid=std_drift_grid,
            avg_entropy_grid=avg_entropy_grid,
            std_entropy_grid=std_entropy_grid
        )

        # Plot conditional distribution across Lagrangian values
        plot_dist_fn(
            distribution_grid=Pmc_grid,
            title='Conditional distribution $P(m|c)$',
            save_path=args.save_dir / 'pmc.pdf' if args.save_dir is not None else None
        )

        # Plot marginal distribution across Lagrangian values
        if args.verbosity >= 1:
            plot_dist_fn(
                distribution_grid=Pm_grid,
                title='Marginal distribution $P(m)$',
                save_path=args.save_dir / 'pm.pdf' if args.save_dir is not None else None
            )
    else:
        # Select output
        if args.outputs == {'drift'}:
            output_type = 'drift'
            units = DRIFT_UNITS
            avg_out_grid = avg_drift_grid
            std_out_grid = std_drift_grid
            parameters = lam_grid[:, 0]
            parameter_name = r'$\lambda$'
        elif args.outputs == {'entropy'}:
            output_type = 'entropy'
            units = ENTROPY_UNITS
            avg_out_grid = avg_entropy_grid
            std_out_grid = std_entropy_grid
            parameters = mu_grid[0]
            parameter_name = r'$\mu$'
        else:
            raise ValueError(f'Outputs "{args.outputs}" not supported.')

        # Remove dimension of size 1 when only using one output
        infos, avg_outs, std_outs, Pmcs, Pms = \
            info_grid.squeeze(), avg_out_grid.squeeze(), std_out_grid.squeeze(), Pmc_grid.squeeze(), Pm_grid.squeeze()

        # Plot average output vs mutual information
        plot_information_and_output(
            infos=infos,
            avg_outs=avg_outs,
            output_type=output_type.title(),
            units=units,
            save_path=args.save_dir / f'{output_type}_vs_information.pdf' if args.save_dir is not None else None
        )

        # Set up distribution plotting function across Lagrangian values
        plot_dist_fn = partial(
            plot_distributions_across_parameters,
            c=c,
            m=m,
            parameters=parameters,
            parameter_name=parameter_name,
            infos=infos,
            avg_outs=avg_outs,
            std_outs=std_outs,
            output_type=output_type,
            units=units
        )

        # Plot conditional distribution across Lagrangian values
        plot_dist_fn(
            distributions=Pmcs,
            title='Conditional distribution $P(m|c)$',
            save_path=args.save_dir / 'pmc.pdf' if args.save_dir is not None else None
        )

        # Plot marginal distribution across Lagrangian values
        if args.verbosity >= 1:
            plot_dist_fn(
                distributions=Pms,
                title='Marginal distribution $P(m)$',
                save_path=args.save_dir / 'pm.pdf' if args.save_dir is not None else None
            )


if __name__ == '__main__':
    run_simulation(Args().parse_args())
