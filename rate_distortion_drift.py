"""
Simulation of E. coli drift and mutual information

References:
    - Micali: Drift and Behavior of E. coli Cells by Micali et al. - Biophysical Journal 2017 (https://www.sciencedirect.com/science/article/pii/S0006349517310755)
    - Taylor: Information and fitness by Taylor, Tishby, and Bialek - arXiv 2007 (https://arxiv.org/abs/0712.4382)
    - Clausznitzer: Chemotactic Response and Adaptation Dynamics in Escherichia coli by Clausznitzer et al. - PLOS Computational Biology 2010 (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000784)
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Constants
CMAP = plt.get_cmap('viridis')
EPS = np.spacing(1)


def set_up_methylation_levels_and_ligand_concentrations() -> Tuple[np.ndarray, float, np.ndarray, float]:
    """
    Sets up methylation levels and ligand concentrations.

    :return: A tuple containing:
               - m (matrix): methylation levels (differing across the rows)
               - dm (float): average difference between methylation levels
               - c (matrix): ligand concentrations (differing across the columns)
               - dc (float): average difference between ligand concentrations
    """
    num_methylation_levels = num_ligand_concentrations = 1000  # Number of levels/concentrations
    mi = np.linspace(8, 0, num_methylation_levels)  # Methylation levels
    ci = np.logspace(-3, 3, num_ligand_concentrations)  # Ligand concentrations (LOG SPACE)
    dm = np.mean(np.diff(mi[::-1]))  # Differences between methylation levels
    dc = np.mean(np.diff(ci))  # Differences between ligand concentrations (intervals in c are not constant since log space)

    c, m = np.meshgrid(ci, mi)  # Mesh grid of ligand concentrations and methylation levels

    return m, dm, c, dc


def compute_drift_and_entropy(c: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    vd = k * A * (1 - A) * dFdc * rel_grad

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
    EP = (dSdty + dSdtm) / 1000  # Micali equation S29 (divide by 1000 to use smaller values)

    return vd, EP


def plot_output(output: np.ndarray,
                c: np.ndarray,
                m: np.ndarray) -> None:
    """
    Plots the output given the ligand concentration and methylation level.

    :param output: A matrix containing the output of interest, which is either drift or entropy production,
                   for different methylation levels (rows) and ligand concentrations (columns).
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    """
    plt.contourf(np.log(c), m, output, levels=64, cmap=CMAP)
    plt.colorbar()
    maxi = np.argmax(output, axis=0)  # Index in each column corresponding to maximum
    plt.plot(np.log(c[0]), m[0, maxi], color='red', label='max')
    plt.legend(loc='upper left')
    plt.xlabel('Ligand $c_0$')
    plt.ylabel('Methylation $m$')
    plt.show()


def set_up_ligand_concentration_distribution(c: np.ndarray,
                                             dc: float,
                                             relative_gradient: float = 0.1) -> np.ndarray:
    """
    Sets up the marginal distribution P(c) over ligand concentrations.

    :param c: A matrix of ligand concentrations (differing across the columns).
    :param dc: The average difference between ligand concentrations.
    :param relative_gradient: The constant relative ligand gradient. (TODO: also try 2)
    :return: A matrix containing the marginal distribution P(c) over ligand concentrations.
    """
    pc = np.exp(-relative_gradient * c)
    Pc = pc / np.sum(pc * dc, keepdims=True, axis=1)

    return Pc


def compute_Pm(Pmc: np.ndarray,
               Pc: np.ndarray,
               dc: float) -> np.ndarray:
    """
    Given the conditional distribution P(m | c), computes the marginal distribution P(m).

    :param Pmc: The conditional distribution P(m | c) over methylation levels given ligand concentrations.
    :param Pc: The marginal distribution P(c) over ligand concentrations.
    :param dc: The average difference between ligand concentrations.
    :return: The marginal distribution P(m) over methylation levels,
             which is a matrix of size (num_methylation_levels, num_ligand_concentrations).
    """
    return np.broadcast_to(np.sum(Pmc * Pc * dc, keepdims=True, axis=1), Pmc.shape)


def compute_Pmc(Pm: np.ndarray,
                output: np.ndarray,
                dm: float,
                lam: float) -> np.ndarray:
    """
    Given the marginal distribution P(m), computes the conditional distribution P(m | c).

    :param Pm: The marginal distribution P(m) over methylation levels.
    :param output: A matrix containing the output of interest, which is either drift or entropy production,
                   for different methylation levels (rows) and ligand concentrations (columns).
    :param dm: The average difference between methylation levels.
    :param lam: The Lagrangian parameter lambda.
    :return: The conditional distribution P(m | c) over methylation levels given ligand concentrations,
             which is a matrix of size (num_methylation_levels, num_ligand_concentrations).
    """
    return np.exp(lam * output) * Pm / np.sum(np.exp(lam * output) * Pm * dm, axis=0)


def plot_conditional_distribution(Pmc: np.ndarray,
                                  c: np.ndarray,
                                  m: np.ndarray,
                                  lam: float) -> None:
    """
    Plots the conditional distribution P(m | c).

    :param Pmc: The conditional distribution P(m | c) over methylation levels given ligand concentrations.
    :param c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param lam: The Lagrangian parameter lambda.
    """
    plt.contourf(np.log(c), m, Pmc, 64, cmap=CMAP)
    plt.colorbar()
    plt.title(rf'Conditional distribution $P(m|c)$ for $\lambda={lam}$')
    plt.ylabel('Methylation level $m$')
    plt.xlabel(r'Ligand concentration $\log(c)$')
    plt.show()


def determine_information_and_output(output: np.ndarray,
                                     Pc: np.ndarray,
                                     dm: float,
                                     dc: float) -> Tuple[List[float], List[float]]:
    """
    Iterates an algorithm to determine the minimum mutual information and maximum mean fitness for different parameters.

    :param output: A matrix containing the output of interest, which is either drift or entropy production,
                   for different methylation levels (rows) and ligand concentrations (columns).
    :param Pc: The marginal distribution P(c) over ligand concentrations.
    :param dm: The average difference between methylation levels.
    :param dc: The average difference between ligand concentrations.
    :return: A tuple with of a list of minimum mutual information values and a list of maximum mean fitness values.
    """
    # TODO: why are the numbers slightly different from Matlab? Is it just differences in numerical precision?
    # TODO: should 0 information have 0 drift instead of 0.04 drift?
    # TODO: numerical issues with error tolerance below 1e-3???

    iter_max = int(2e4)  # Maximum number of iterations (10)
    error_tol = 1e-1  # Error tolerance for convergence (1e-4, 1e-5)
    Imins, outmaxes = [], []
    for lam in np.logspace(0, 1, 10):  # (1, 2, 10)
        print(f'Lambda = {lam:.2f}')

        # Initial guess for marginal distribution P(m) over methylation levels
        Pm = np.ones(Pc.shape)

        # Normalize P(m)
        Pm = Pm / np.sum(Pm * dm, axis=0)

        # Initial guess for conditional distribution P(m | c) over methylation levels given ligand concentrations
        Pmc = compute_Pmc(Pm=Pm, output=output, dm=dm, lam=lam)

        for i in range(iter_max):
            # Save previous P(m)
            Pm_old = Pm

            # Compute new P(m)
            Pm = compute_Pm(Pmc=Pmc, Pc=Pc, dc=dc)

            # Compute new P(m, c)
            Pmc = compute_Pmc(Pm=Pm, output=output, dm=dm, lam=lam)

            # Extract one column of Pm and Pm_old to represent new P(m) and old P(m) since all columns are identical
            Pm_col, Pm_old_col = Pm[:, 0], Pm_old[:, 0]

            # If difference between new P(m) and old P(m) is below an error tolerance, then algorithm has converged
            if np.linalg.norm(Pm_col - Pm_old_col) * dm <= error_tol:  # TODO: why multiply by dm outside of norm?
                print(f'Converged for lambda = {lam:.2f} after {i + 1} iterations')

                # Compute the minimum mutual information (Taylor equation 1)
                Imin = np.sum(dc * Pc * np.sum(dm * Pmc * np.log2(EPS + Pmc / (EPS + Pm)), axis=0), axis=1)

                # Compute maximum mean output, which is either drift or entropy production (Taylor equation 4)
                outmax = np.sum(dc * Pc * np.sum(dm * Pmc * output, axis=0), axis=1)

                # Save Imin and outmax (only include 0th element since lal elements are the same)
                Imins.append(Imin[0])
                outmaxes.append(outmax[0])
                break

    return Imins, outmaxes, Pmc, lam


def plot_information_and_output(Imins: List[float],
                                outmaxes: List[float]) -> None:
    """
    Plot the maximum mean output vs the minimum mutual information.

    :param Imins: A list of minimum mutual information values.
    :param outmaxes: A list of maximum mean output values.
    """
    # Plot mutual information vs output
    plt.plot(Imins, outmaxes, 'x')
    plt.title('Mean Output vs Mutual Information')
    plt.ylabel('Mean output')
    plt.xlabel('Mutual Information $I(m; c_0)$ (bits)')
    plt.show()


def run_simulation() -> None:
    """Runs the rate distortion simulation."""
    # Set up the methylation levels and ligand concentrations
    m, dm, c, dc = set_up_methylation_levels_and_ligand_concentrations()

    # Compute drift and entropy
    vd, EP = compute_drift_and_entropy(c=c, m=m)

    # Select output
    output = vd

    # Plot output
    plot_output(output=output, c=c, m=m)

    # Set up marginal distribution over ligand concentrations P(c)
    Pc = set_up_ligand_concentration_distribution(c=c, dc=dc)

    # Determine minimum mutual information and maximum mean output for multiple parameter values
    Imins, outmaxes = determine_information_and_output(output=output, Pc=Pc, dm=dm, dc=dc)

    # Plot mutual information and mean output
    plot_information_and_output(Imins=Imins, outmaxes=outmaxes)

    # Conditional distribution for final lambda value
    # plot_conditional_distribution(Pmc=Pmc, c=c, m=m, lam=lam)


if __name__ == '__main__':
    run_simulation()
