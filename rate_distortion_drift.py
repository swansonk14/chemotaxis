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
from pathlib import Path
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid as integrate
from tap import Tap
from tqdm import tqdm, trange


class Args(Tap):
    output_type: Literal['drift', 'entropy', 'drift-entropy'] = 'drift'
    """The output whose mutual information will be computed."""
    iter_max: int = 100
    """Maximum number of iterations of the algorithm."""
    lambda_min: float = 1.0
    """Minimum value of lambda in log space (i.e., min lambda = 10^{lambda_min})."""
    lambda_max: float = 3.5
    """Maximum value of lambda in log space (i.e., min lambda = 10^{lambda_max})."""
    lambda_num: int = 9
    """Number of lambda values between lambda_min and lambda_max."""
    mu: float = 0.01
    """Lagrangian mu applied to the entropy. Only relevant for output_type == 'drift-entropy'."""
    ligand_gradient: float = 0.1
    """The relative gradient of the ligand concentration."""
    verbosity: Literal[0, 1, 2] = 1
    """Verbosity level. Higher means more verbose."""
    save_dir: Path = None
    """Directory where plots and arguments will be saved (if None, displayed instead)."""

    def process_args(self) -> None:
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)


# Constants
CMAP = plt.get_cmap('viridis')
EPS = np.spacing(1)
DPI = 300


def set_up_methylation_levels_and_ligand_concentrations() -> Tuple[np.ndarray, np.ndarray]:
    """
    Sets up methylation levels and ligand concentrations.

    :return: A tuple containing:
               - m (matrix): methylation levels (differing across the rows)
               - c (matrix): ligand concentrations (differing across the columns)
    """
    num_methylation_levels = num_ligand_concentrations = 1000  # Number of levels/concentrations
    mi = np.linspace(0, 8, num_methylation_levels)  # Methylation levels TODO: (0, 15)
    ci = np.logspace(-3, 3, num_ligand_concentrations)  # Ligand concentrations (log space) TODO: (-3, 6)

    c, m = np.meshgrid(ci, mi)  # Mesh grid of ligand concentrations and methylation levels

    return m, c


def compute_drift_and_entropy_production(c: np.ndarray,
                                         m: np.ndarray,
                                         ligand_gradient: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the drift and entropy production.

    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param ligand_gradient: The relative gradient of the ligand concentration.
    :return: A tuple consisting of a matrix of drift values and a matrix of entropy production values.
    """
    # Parameters (Micali table S1)
    N = 5  # Cooperative receptor number (paper uses 13, range[5; 13])
    va = 1 / 3  # Fraction of Tar receptors (paper uses 1 / 3)
    vs = 2 / 3  # Fraction of Tsr receptors (paper uses 2 / 3)
    Kaon = 0.5  # Active receptors dissociation constant Tar (mM, paper uses 1.0)
    Kson = 100000  # Active receptors dissociation constant Tsr (mM, paper uses 1E6)
    Kaoff = 0.02  # Inactive receptors dissociation constant Tar (mM, paper uses 0.03)
    Ksoff = 100  # Inactive receptors dissociation constant Tsr (mM, paper uses 100)
    YT = 9.7  # Total concentration of CheY (muM, paper uses 7.9, range [6; 9.7])
    k = 1  # Approximation of the susceptibility function.

    # Phosphorylation rate parameters (Micali equation S28)
    nu = 0.01  # Proportionality constant between forward and backward rates (paper says << 1)

    kyp = 100  # Rate constant k_y^+ for reaction CheY + ATP --> CheY_p + ADP (muM^{-1}s^{-1})
    kym = nu * kyp  # Rate constant k_y^- for reaction CheY_p + ADP --> CheY + ATP (muM^{-1}s^{-1})

    kzp = 30  # Rate constant k_z^+ for reaction CheY_p + ADP --> CheY + ADP + Pi (s^{-1})
    kzm = nu * kzp  # Rate constant k_z^- for reaction CheY + ADP + Pi --> CheY_p + ADP (s^{-1})

    # Methylation rate parameters (Micali equation S29)
    kRp = 0.0069  # Rate constant k_R^+ for reaction [m]_0 + SAM --> [m + 1]_0 + SAH (s^{-1})
    kRm = nu * kRp  # Rate constant k_R^- for reaction [m + 1]_0 + SAH --> [m]_0 + SAM (s^{-1})
    kBp = 0.12  # Rate constant k_B^+ for reaction [m + 1]_1 + H2O --> [m]_1 + CH3OH (s^{-1})
    kBm = nu * kBp  # Rate constant k_B^+ for reaction [m]_1 + CH3OH --> [m + 1]_1 + H2O (s^{-1})
    mT = 1e4  # Total number of methylation sites

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

    # Drift velocity (Micali equation 1)
    drift = k * A * (1 - A) * dFdc * ligand_gradient

    # Concentration of phosphorylated CheY (CheY_p)
    # from CheY + ATP --> CheY_p + ADP (kyp rate constant) and CheY + ADP + Pi --> CheY_p + ADP (kzm rate constant)
    Yp = (kyp * A + kzm) * YT / ((kyp + kym) * A + kzp + kzm)  # TODO: Micali equation S4???

    # Entropy production of the phosphorylation dynamics (Micali equation S28 without Boltzmann constant kb)
    dSdty = (kyp * (YT - Yp) - kym * Yp) * A * np.log((kyp * (YT - Yp)) / (kym * Yp)) + \
            (kzp * Yp - kzm * (YT - Yp)) * np.log((kzp * Yp) / (kzm * (YT - Yp)))

    # Entropy production of the methylation dynamics (Micali equation S29 without Boltzmann constant kb)
    dSdtm = (kRp - kRm) * (1 - A) * mT * np.log(kRp / kRm) + \
            (kBp - kBm) * A**3 * mT * np.log(kBp / kBm)

    # Entropy production of the phosphorylation and methylation dynamics
    entropy_production = (dSdty + dSdtm) / 1000  # Micali equation S29 (divide by 1000 to use smaller values)

    return drift, entropy_production


def plot_output(output: np.ndarray,
                output_type: str,
                c: np.ndarray,
                m: np.ndarray,
                plot_max: bool = False,
                save_path: Path = None) -> None:
    """
    Plots the output given the ligand concentration and methylation level.

    :param output: A matrix containing the output of interest for different methylation levels (rows)
                   and ligand concentrations (columns).
    :param output_type: The name of the type of output.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param plot_max: Whether to plot the maximum y value for each x value.
    :param save_path: Path where the plot will be saved (if None, displayed instead).
    """
    plt.clf()
    plt.contourf(np.log(c), m, output, levels=64, cmap=CMAP)
    plt.colorbar()

    if plot_max:
        maxi = np.argmax(output, axis=0)  # Index in each column corresponding to maximum
        plt.plot(np.log(c[0]), m[maxi, 0], color='red', label='max')

    plt.title(f'{output_type.title()} for given ligand concentration and methylation level')
    plt.legend(loc='upper left')
    plt.xlabel(r'Ligand concentration $\log(c)$')
    plt.ylabel('Methylation level $m$')

    if save_path is not None:
        plt.savefig(save_path, dpi=DPI)
    else:
        plt.show()


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
                exp_lam_output: np.ndarray) -> np.ndarray:
    """
    Given the marginal distribution P(m), computes the conditional distribution P(m | c).

    :param Pm: The marginal distribution P(m) over methylation levels.
    :param m: A matrix of methylation levels (differing across the rows).
    :param exp_lam_output: A matrix containing exp(lam * output).
    :return: The conditional distribution P(m | c) over methylation levels given ligand concentrations,
             which is a matrix of size (num_methylation_levels, num_ligand_concentrations).
    """
    pmc = Pm * exp_lam_output
    Pmc = pmc / integrate(pmc, m, axis=0)

    return Pmc


def plot_information_output_and_objective(Is: List[float],
                                          outs: List[float],
                                          objectives: List[float],
                                          lam: float,
                                          save_path: Path = None) -> None:
    """
    Plots the mutual information, output, and objective function across iterations.

    :param Is: A list of mutual information values across iterations.
    :param outs: A list of output values across iterations.
    :param objectives: A list of objective function values across iterations.
    :param lam: Lagrangian parameter lambda.
    :param save_path: Path where the plot will be saved (if None, displayed instead).
    """
    plt.clf()
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True)
    ax0.scatter(np.arange(len(Is)), Is, color='red', label='I', s=3)
    ax1.scatter(np.arange(len(outs)), outs, color='blue', label='out', s=3)
    ax2.scatter(np.arange(len(objectives)), objectives, color='green', label='objective', s=3)
    ax0.legend()
    ax1.legend()
    ax2.legend()
    plt.xlabel('Iteration')
    ax0.set_title(rf'I, out, and objective function for $\lambda = {lam:.2f}$')

    if save_path is not None:
        plt.savefig(save_path, dpi=DPI)
    else:
        plt.show()


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


def determine_information_and_output(output: np.ndarray,
                                     Pc: np.ndarray,
                                     c: np.ndarray,
                                     m: np.ndarray,
                                     iter_max: int,
                                     lambda_min: float,
                                     lambda_max: float,
                                     lambda_num: int,
                                     verbosity: int,
                                     save_dir: Path = None) -> Tuple[List[float],
                                                                     List[float],
                                                                     List[np.ndarray],
                                                                     List[np.ndarray],
                                                                     np.ndarray]:
    """
    Iterates an algorithm to determine the minimum mutual information and maximum mean fitness for different parameters.

    :param output: A matrix containing the output of interest
                   for different methylation levels (rows) and ligand concentrations (columns).
    :param Pc: The marginal distribution P(c) over ligand concentrations.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param iter_max: Maximum number of iterations of the algorithm.
    :param lambda_min: Minimum value of lambda in log space (i.e., min lambda = 10^{lambda_min}).
    :param lambda_max: Maximum value of lambda in log space (i.e., min lambda = 10^{lambda_max}).
    :param lambda_num: Number of lambda values between lambda_min and lambda_max.
    :param verbosity: Verbosity level. Higher means more verbose.
    :param save_dir: Directory where the plot will be saved (if None, displayed instead).
    :return: A tuple containing:
               - Imins (List[float]): a list of minimum mutual information values
               - outmaxes (List[float]): a list of maximum mean output values
               - Pmcs (List[np.ndarray]): a list of  conditional distributions P(m | c)
               - Pms (List[np.ndarray]): a list of marginal distributions P(m)
               - lams (np.ndarray): a numpy array of lambda values
    """
    Imins, outmaxes, Pmcs, Pms = [], [], [], []
    lams = np.logspace(lambda_min, lambda_max, lambda_num)
    for lam in lams:
        print(f'Lambda = {lam:.2f}')
        # Keep track of I, out, and objective function across iterations
        Is, outs, objectives = [], [], []

        # Initial guess for marginal distribution P(m) over methylation levels
        Pm = np.ones(Pc.shape)

        # Normalize P(m)
        Pm = Pm / integrate(Pm, m, axis=0)

        # Precompute exp(lam * output)
        exp_lam_output = np.exp(lam * output)

        # Initial guess for conditional distribution P(m | c) over methylation levels given ligand concentrations
        Pmc = compute_Pmc(Pm=Pm, m=m, exp_lam_output=exp_lam_output)

        for _ in trange(iter_max):
            # Compute new P(m)
            Pm = compute_Pm(Pmc=Pmc, Pc=Pc, c=c)

            # Compute new P(m | c)
            Pmc = compute_Pmc(Pm=Pm, m=m, exp_lam_output=exp_lam_output)

            # Keep track of I, out, and objective function across iterations
            if verbosity >= 2:
                I = compute_mutual_information(Pmc=Pmc, Pc=Pc, Pm=Pm, c=c, m=m)
                out = compute_average_output(output=output, Pmc=Pmc, Pc=Pc, c=c, m=m)
                objective = I - lam * out

                Is.append(I)
                outs.append(out)
                objectives.append(objective)

        # Compute the minimum mutual information (Taylor equation 1)
        Imin = compute_mutual_information(Pmc=Pmc, Pc=Pc, Pm=Pm, c=c, m=m)

        # Compute maximum mean output (Taylor equation 4)
        outmax = compute_average_output(output=output, Pmc=Pmc, Pc=Pc, c=c, m=m)

        # Save Imin, outmax, and Pmc (only include 0th element since all elements are the same)
        Imins.append(Imin)
        outmaxes.append(outmax)
        Pmcs.append(Pmc)
        Pms.append(Pm)

        # Plot I, out, and objective function across iterations
        if verbosity >= 2:
            plot_information_output_and_objective(
                Is=Is,
                outs=outs,
                objectives=objectives,
                lam=lam,
                save_path=save_dir / f'convergence_lambda={lam:.2f}.png'
            )

    return Imins, outmaxes, Pmcs, Pms, lams


def plot_information_and_output(Imins: List[float],
                                outmaxes: List[float],
                                output_type: str,
                                save_path: Path = None) -> None:
    """
    Plot the maximum mean output vs the minimum mutual information.

    :param Imins: A list of minimum mutual information values.
    :param outmaxes: A list of maximum mean output values.
    :param output_type: The name of the type of output.
    :param save_path: Path where the plot will be saved (if None, displayed instead).
    """
    plt.clf()
    plt.plot(Imins, outmaxes, 'x')
    plt.title(f'{output_type.title()} vs Mutual Information')
    plt.ylabel(output_type.title())
    plt.xlabel('Mutual Information $I(m; c)$ (bits)')

    if save_path is not None:
        plt.savefig(save_path, dpi=DPI)
    else:
        plt.show()


def plot_distributions_across_lambdas(distributions: List[np.ndarray],
                                      c: np.ndarray,
                                      m: np.ndarray,
                                      lams: np.ndarray,
                                      Imins: List[float],
                                      outmaxes: List[float],
                                      output_type: str,
                                      title: str,
                                      save_path: Path = None) -> None:
    """
    Plots the distributions over methylation levels and ligand concentrations across lambda values.

    :param distributions: A list of probability distributions over methylation levels
                          and ligand concentrations across lambda values.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param lams: A list of values of the Lagrangian parameter lambda.
    :param Imins: A list of minimum mutual information values.
    :param outmaxes: A list of maximum mean output values.
    :param output_type: The name of the type of output.
    :param title: The title of the plot.
    :param save_path: Path where the plot will be saved (if None, displayed instead).
    """
    log_c = np.log(c)
    size = int(np.ceil(np.sqrt(len(lams))))  # Number of rows/columns in a square that can hold all the plots

    plt.clf()
    fig, axes = plt.subplots(nrows=size, ncols=size, figsize=2.25 * np.array([6.4, 4.8]))
    axes = [axes] if size == 1 else axes.flat

    for ax, distribution, lam, Imin, outmax in tqdm(zip(axes, distributions, lams, Imins, outmaxes), total=len(lams)):
        im = ax.contourf(log_c, m, distribution, levels=64, cmap=CMAP)
        fig.colorbar(im, ax=ax)
        ax.title.set_text(rf'$\lambda={lam:.2f}, I={Imin:.2f}, {output_type}={outmax:.2f}$')

    fig.suptitle(title)
    fig.text(0.04, 0.5, 'Methylation level $m$', va='center', rotation='vertical')  # y label
    fig.text(0.5, 0.04, r'Ligand concentration $\log(c)$', ha='center')  # x label

    if save_path is not None:
        plt.savefig(save_path, dpi=DPI)
    else:
        plt.show()


def run_simulation(args: Args) -> None:
    """Runs the rate distortion simulation."""
    # Save arguments
    if args.save_dir is not None:
        args.save(args.save_dir / 'args.json')

    # Set up the methylation levels and ligand concentrations
    m, c = set_up_methylation_levels_and_ligand_concentrations()

    # Compute drift and entropy
    drift, entropy_production = compute_drift_and_entropy_production(
        c=c,
        m=m,
        ligand_gradient=args.ligand_gradient
    )

    # Select output
    if args.output_type == 'drift':
        output = drift
    elif args.output_type == 'entropy':
        output = -entropy_production
    elif args.output_type == 'drift-entropy':
        output = drift - args.mu * entropy_production
    else:
        raise ValueError(f'Output type "{args.output_type}" is not supported.')

    # Plot output
    if args.verbosity >= 1:
        plot_output(
            output=output,
            output_type=args.output_type,
            c=c,
            m=m,
            plot_max=True,
            save_path=args.save_dir / f'{args.output_type}.png' if args.save_dir is not None else None
        )

    # Set up marginal distribution over ligand concentrations P(c)
    Pc = set_up_ligand_concentration_distribution(c=c, ligand_gradient=args.ligand_gradient)

    # Plot P(c)
    if args.verbosity >= 1:
        plot_output(
            output=Pc,
            output_type='$P(c)$',
            c=c,
            m=m,
            save_path=args.save_dir / 'pc.png' if args.save_dir is not None else None
        )

    # Determine minimum mutual information and maximum mean output for multiple parameter values
    Imins, outmaxes, Pmcs, Pms, lams = determine_information_and_output(
        output=output,
        Pc=Pc,
        m=m,
        c=c,
        iter_max=args.iter_max,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        lambda_num=args.lambda_num,
        verbosity=args.verbosity,
        save_dir=args.save_dir
    )

    # Plot mutual information and mean output
    plot_information_and_output(
        Imins=Imins,
        outmaxes=outmaxes,
        output_type=args.output_type,
        save_path=args.save_dir / f'{args.output_type}_vs_information.png' if args.save_dir is not None else None
    )

    # Plot conditional distribution for all lambda values
    plot_distributions_across_lambdas(
        distributions=Pmcs,
        c=c,
        m=m,
        lams=lams,
        Imins=Imins,
        outmaxes=outmaxes,
        output_type=args.output_type,
        title='Conditional distribution $P(m|c)$',
        save_path=args.save_dir / 'pmc.png' if args.save_dir is not None else None
    )

    # Plot marginal distribution for all lambda values
    if args.verbosity >= 1:
        plot_distributions_across_lambdas(
            distributions=Pms,
            c=c,
            m=m,
            lams=lams,
            Imins=Imins,
            outmaxes=outmaxes,
            output_type=args.output_type,
            title='Marginal distribution $P(m)$',
            save_path=args.save_dir / 'pm.png' if args.save_dir is not None else None
        )


if __name__ == '__main__':
    run_simulation(Args().parse_args())
