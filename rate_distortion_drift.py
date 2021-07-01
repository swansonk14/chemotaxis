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
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid as integrate  # TODO: change to function integration instead of sample integration?
from tap import Tap
from tqdm import tqdm, trange


class Args(Tap):
    output_type: Literal['drift', 'entropy', 'drift - entropy'] = 'drift'  # The output whose mutual information will be computed.
    verbose: bool = False  # Whether to print/plot additional information.


# Constants
CMAP = plt.get_cmap('viridis')
EPS = np.spacing(1)


def set_up_methylation_levels_and_ligand_concentrations() -> Tuple[np.ndarray, np.ndarray]:
    """
    Sets up methylation levels and ligand concentrations.

    :return: A tuple containing:
               - m (matrix): methylation levels (differing across the rows)
               - c (matrix): ligand concentrations (differing across the columns)
    """
    num_methylation_levels = num_ligand_concentrations = 1000  # Number of levels/concentrations
    mi = np.linspace(0, 8, num_methylation_levels)  # Methylation levels
    ci = np.logspace(-3, 3, num_ligand_concentrations)  # Ligand concentrations (LOG SPACE)

    c, m = np.meshgrid(ci, mi)  # Mesh grid of ligand concentrations and methylation levels

    return m, c


def compute_drift_and_entropy_production(c: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the drift and entropy production.

    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :return: A tuple consisting of a matrix of drift values and a matrix of entropy production values.
    """
    # Parameters (Micali table S1)
    N = 5  # Cooperative receptor number (paper uses 13, range[5; 13])
    va = 1 / 3  # Fraction of Tar receptors (paper uses 1 / 3)
    vs = 2 / 3  # Fraction of Tsr receptors (paper uses 1 / 3)
    Kaon = 0.5  # Active receptors dissociation constant Tar (mM, paper uses 1.0)
    Kson = 100000  # Active receptors dissociation constant Tsr (mM, paper uses 1E6)
    Kaoff = 0.02  # Inactive receptors dissociation constant Tar (mM, paper uses 0.03)
    Ksoff = 100  # Inactive receptors dissociation constant Tsr (mM, paper uses 100)
    YT = 9.7  # Total concentration of CheY (muM, paper uses 7.9, range [6; 9.7])
    k = 1  # TODO: what is this? susceptibility? motor dissociation constant?

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

    # Ligand (local) gradient steepness
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

    # Drift velocity (Micali equation 1) TODO: why k * A instead of function K(<A>)?
    drift = k * A * (1 - A) * dFdc * rel_grad

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
                m: np.ndarray) -> None:
    """
    Plots the output given the ligand concentration and methylation level.

    :param output: A matrix containing the output of interest, which is either drift or entropy production,
                   for different methylation levels (rows) and ligand concentrations (columns).
    :param output_type: The name of the type of output (either "drift" or "entropy").
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    """
    plt.contourf(np.log(c), m, output, levels=64, cmap=CMAP)
    plt.colorbar()
    maxi = np.argmax(output, axis=0)  # Index in each column corresponding to maximum
    plt.plot(np.log(c[0]), m[maxi, 0], color='red', label='max')
    plt.title(f'{output_type.title()} for given ligand concentration and methylation level')
    plt.legend(loc='upper left')
    plt.xlabel(r'Ligand concentration $\log(c)$')
    plt.ylabel('Methylation level $m$')
    plt.show()


def set_up_ligand_concentration_distribution(c: np.ndarray, relative_gradient: float = 0.1) -> np.ndarray:
    """
    Sets up the marginal distribution P(c) over ligand concentrations.

    :param c: A matrix of ligand concentrations (differing across the columns).
    :param relative_gradient: The constant relative ligand gradient. (TODO: also try 2)
    :return: A matrix containing the marginal distribution P(c) over ligand concentrations.
    """
    pc = np.exp(-relative_gradient * c)
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


def determine_information_and_output(output: np.ndarray,
                                     Pc: np.ndarray,
                                     c: np.ndarray,
                                     m: np.ndarray,
                                     verbose: bool = False) -> Tuple[List[float], List[float], List[np.ndarray], np.ndarray]:
    """
    Iterates an algorithm to determine the minimum mutual information and maximum mean fitness for different parameters.

    :param output: A matrix containing the output of interest, which is either drift or entropy production,
                   for different methylation levels (rows) and ligand concentrations (columns).
    :param Pc: The marginal distribution P(c) over ligand concentrations.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param verbose: Whether to print/plot additional information.
    :return: A tuple containing:
               - Imins (List[float]): a list of minimum mutual information values
               - outmaxes (List[float]): a list of maximum mean output values
               - Pmcs (List[np.ndarray]): a list of  conditional distributions P(m | c)
               - lams (np.ndarray): a numpy array of lambda values
    """
    # TODO: why are the numbers slightly different from Matlab? Is it just differences in numerical precision?
    # TODO: should 0 information have 0 drift instead of 0.04 drift?
    # TODO: numerical issues with error tolerance below 1e-3???

    iter_max = 50  # Maximum number of iterations (10)
    # TODO: change convergence to be based on objective function rather than P(m)?
    # error_tol = 1e-2  # Error tolerance for convergence (1e-4, 1e-5)  TODO: remove?
    Imins, outmaxes, Pmcs = [], [], []
    lams = np.logspace(-1, 3, 9)    # (0, 1, 10)
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

        for i in trange(iter_max):
            # Save previous P(m)  TODO: remove?
            # Pm_old = Pm

            # Compute new P(m)
            Pm = compute_Pm(Pmc=Pmc, Pc=Pc, c=c)

            # Compute new P(m | c)
            Pmc = compute_Pmc(Pm=Pm, m=m, exp_lam_output=exp_lam_output)

            # Keep track of I, out, and objective function across iterations
            if verbose:
                Imin = integrate(Pc * integrate(Pmc * np.log2(EPS + Pmc / (EPS + Pm)), m, axis=0), c, axis=1)
                outmax = integrate(Pc * integrate(Pmc * output, m, axis=0), c, axis=1)
                objective = Imin - lam * outmax

                Is.append(Imin[0])
                outs.append(outmax[0])
                objectives.append(objective[0])

            # Extract one column of Pm and Pm_old to represent new P(m) and old P(m) since all columns are identical  TODO: remove?
            # Pm_col, Pm_old_col = Pm[:, 0], Pm_old[:, 0]

            # If difference between new P(m) and old P(m) is below an error tolerance, then algorithm has converged  TODO: remove
            # if np.linalg.norm(Pm_col - Pm_old_col) <= error_tol or i == iter_max - 1:

            if i == iter_max - 1:
                print(f'Converged for lambda = {lam:.2f} after {i + 1} iterations')

                # Compute the minimum mutual information (Taylor equation 1)
                Imin = integrate(Pc * integrate(Pmc * np.log2(EPS + Pmc / (EPS + Pm)), m, axis=0), c, axis=1)

                # Compute maximum mean output, which is either drift or entropy production (Taylor equation 4)
                outmax = integrate(Pc * integrate(Pmc * output, m, axis=0), c, axis=1)

                # Save Imin, outmax, and Pmc (only include 0th element since all elements are the same)
                Imins.append(Imin[0])
                outmaxes.append(outmax[0])
                Pmcs.append(Pmc)
                break

        # Plot I, out, and objective function across iterations
        if verbose:
            plt.scatter(np.arange(len(Is)), Is, color='red', label='I', s=3)
            plt.scatter(np.arange(len(outs)), outs, color='blue', label='out', s=3)
            plt.scatter(np.arange(len(objectives)), objectives, color='green', label='objective', s=3)
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.title(rf'I, out, and objective function for $\lambda = {lam:.2f}$')
            plt.show()

    return Imins, outmaxes, Pmcs, lams


def plot_information_and_output(Imins: List[float],
                                outmaxes: List[float],
                                output_type: str) -> None:
    """
    Plot the maximum mean output vs the minimum mutual information.

    :param Imins: A list of minimum mutual information values.
    :param outmaxes: A list of maximum mean output values.
    :param output_type: The name of the type of output (either "drift" or "entropy").
    """
    # Plot mutual information vs output
    plt.plot(Imins, outmaxes, 'x')
    plt.title(f'{output_type.title()} vs Mutual Information')
    plt.ylabel(output_type.title())
    plt.xlabel('Mutual Information $I(m; c)$ (bits)')
    plt.show()


def plot_conditional_distributions(Pmcs: List[np.ndarray],
                                   c: np.ndarray,
                                   m: np.ndarray,
                                   lams: np.ndarray,
                                   Imins: List[float],
                                   outmaxes: List[float],
                                   output_type: str) -> None:
    """
    Plots the conditional distribution P(m | c).

    :param Pmcs: A list of conditional distributions P(m | c) over methylation levels given ligand concentrations.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param lams: A list of values of the Lagrangian parameter lambda.
    :param Imins: A list of minimum mutual information values.
    :param outmaxes: A list of maximum mean output values.
    :param output_type: The name of the type of output (either "drift" or "entropy").
    """
    log_c = np.log(c)
    size = int(np.ceil(np.sqrt(len(lams))))  # Number of rows/columns in a square that can hold all the plots

    fig, axes = plt.subplots(nrows=size, ncols=size)
    axes = [axes] if size == 1 else axes.flat

    for ax, Pmc, lam, Imin, outmax in tqdm(zip(axes, Pmcs, lams, Imins, outmaxes), total=len(lams)):
        im = ax.contourf(log_c, m, Pmc, levels=64, cmap=CMAP)
        fig.colorbar(im, ax=ax)
        ax.title.set_text(rf'$\lambda={lam:.2f}, I={Imin:.2f}, {output_type}={outmax:.2f}$')

    fig.suptitle('Conditional distribution $P(m|c)$')
    fig.text(0.04, 0.5, 'Methylation level $m$', va='center', rotation='vertical')  # y label
    fig.text(0.5, 0.04, r'Ligand concentration $\log(c)$', ha='center')  # x label
    plt.show()


def run_simulation(args: Args) -> None:
    """Runs the rate distortion simulation."""
    # Set up the methylation levels and ligand concentrations
    m, c = set_up_methylation_levels_and_ligand_concentrations()

    # Compute drift and entropy
    drift, entropy_production = compute_drift_and_entropy_production(c=c, m=m)

    # Select output
    if args.output_type == 'drift':
        output = drift
    elif args.output_type == 'entropy':
        output = entropy_production  # TODO: should this be negative so that we're minimizing entropy?
    elif args.output_type == 'drift - entropy':
        output = drift - 0.05 * entropy_production
    else:
        raise ValueError(f'Output type "{args.output_type}" is not supported.')

    # Plot output
    plot_output(output=output, output_type=args.output_type, c=c, m=m)

    # Set up marginal distribution over ligand concentrations P(c)
    Pc = set_up_ligand_concentration_distribution(c=c)

    # Determine minimum mutual information and maximum mean output for multiple parameter values
    Imins, outmaxes, Pmcs, lams = determine_information_and_output(output=output, Pc=Pc, m=m, c=c, verbose=args.verbose)

    # Plot mutual information and mean output
    plot_information_and_output(Imins=Imins, outmaxes=outmaxes, output_type=args.output_type)

    # Conditional distribution for final lambda value
    plot_conditional_distributions(
        Pmcs=Pmcs,
        c=c,
        m=m,
        lams=lams,
        Imins=Imins,
        outmaxes=outmaxes,
        output_type=args.output_type
    )


if __name__ == '__main__':
    run_simulation(Args().parse_args())
