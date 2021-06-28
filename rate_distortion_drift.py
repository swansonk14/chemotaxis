"""
Simulation of E. coli drift and mutual information

References:
    - Micali: Drift and Behavior of E. coli Cells by Micali et al. - Biophysical Journal 2017 (https://www.sciencedirect.com/science/article/pii/S0006349517310755)
    - Taylor: Information and fitness by Taylor, Tishby, and Bialek - arXiv 2007 (https://arxiv.org/abs/0712.4382)
    - Clausznitzer: Chemotactic Response and Adaptation Dynamics in Escherichia coli by Clausznitzer et al. - PLOS Computational Biology 2010 (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000784)
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up parameters and equations
cmap = plt.get_cmap('viridis')
eps = np.spacing(1)

# Methylation and ligand levels
L = 1000  # Number of levels
mi = np.linspace(8, 0, L)  # Methylation levels
ci = np.logspace(-3, 3, L)  # Ligand levels (LOG SPACE)
dm = np.mean(np.diff(mi[::-1]))  # Differences between methylation levels
dc = np.mean(np.diff(ci))  # Differences between ligand levels (note: intervals in c are not constant due to log space)

[c, m] = np.meshgrid(ci, mi)  # Mesh grid of ligand and methylation levels

# Fitness function - estimated

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

# Choose an output(drift vd or entropy production EP)
# output = EP
output = vd

# Input - output curve of methylation and ligand concentration vs output
"""
plt.contourf(np.log(c), m, output, levels=64, cmap=cmap)
plt.colorbar()
maxi = np.argmax(output, axis=0)  # Index in each column corresponding to maximum
plt.plot(np.log(ci), mi[maxi], color='red', label='max')
plt.legend(loc='upper left')
plt.xlabel('Ligand $c_0$')
plt.ylabel('Methylation $m$')
plt.show()
"""

# Probability distribution P(c) over external states of ligand concentration c
r = 0.1  # Constant relative ligand gradient (2)
Pc = np.exp(-r * c) / np.sum(np.exp(-r * c) * dc, axis=0)  # TODO: is this supposed to be nearly all 0.000999??? also is the normalization constant right? why dc or is that to make it an integral? why not use actually diffs rather than mean diff of log space?
# Pc = np.exp(-r * c) / np.sum(np.exp(-r * c) * dc, keepdims=True, axis=1)  # TODO: check whether axis=1 is right and check integration using dc

# Functions to iterate


def Eqn2(Pmc: np.ndarray) -> np.ndarray:
    """
    Given the conditional distribution P(m | c), computes the marginal distribution P(m).

    :param Pmc: The conditional distribution P(m | c) over methylation levels given ligand concentrations.
    :return: The marginal distribution P(m) over methylation levels,
             which is a matrix of size (num_methylation_levels, 1).
    """
    return np.sum(Pmc * Pc * dc, axis=1)


def Eqn5(Pm: np.ndarray, lam: float) -> np.ndarray:
    """
    Given the marginal distribution P(m), computes the conditional distribution P(m | c).

    :param Pm: The marginal distribution P(m) over methylation levels.
    :param lam: The Lagrangian parameter lambda.
    :return: The conditional distribution P(m | c) over methylation levels given ligand concentrations,
             which is a matrix of size (num_methylation_levels, num_ligand_concentrations).
    """
    return np.exp(lam * output) * Pm / np.sum(np.exp(lam * output) * Pm * dm, axis=0)


# Iterate mutual information algorithm

# TODO: why are the numbers slightly different from Matlab? Is it just differences in numerical precision?
# TODO: should 0 information have 0 drift instead of 0.04 drift?

iter_max = int(2e4)  # Maximum number of iterations (10)
error_tol = 1e-1  # Error tolerance for convergence (-4, -5)
for lam in np.logspace(0, 1, 10):  # (1, 2, 10)
    print(f'Lambda = {lam:.2f}')

    # Initial guess for marginal distribution P(m) over methylation levels
    Pm = np.ones(m.shape)

    # Normalize P(m)
    Pm = Pm / np.sum(Pm * dm, axis=0)  # TODO: check axis

    # Initial guess for conditional distribution P(m | c) over methylation levels given ligand concentrations
    Pmc = Eqn5(Pm, lam)  # columns not normalised

    for i in range(iter_max):
        # Save previous P(m)
        Pm_old = Pm

        # Compute new P(m)
        Pm = Eqn2(Pmc)

        # Compute new P(m, c)
        Pmc = Eqn5(Pm, lam)

        # If difference between new P(m) and old P(m) is below an error tolerance, then algorithm has converged
        if np.linalg.norm(Pm - Pm_old) * dm <= error_tol:
            print(f'Converged for lambda = {lam:.2f} after {i + 1} iterations')

            # Compute the minimum mutual information (Taylor equation 1)
            Imin = np.sum(dc * Pc * np.sum(dm * Pmc * np.log2(eps + Pmc / (eps + Pm)), axis=0), axis=1)  # TODO: check axis

            # Compute maximum mean output, which is either drift or entropy production (Taylor equation 4)
            outmax = np.sum(dc * Pc * np.sum(dm * Pmc * output, axis=0), axis=1)  # TODO: check axis

            # Plot mutual information vs output
            plt.plot(Imin, outmax, 'x')  # TODO: should this be 1000 of the same number?
            break

plt.ylabel('Mean output')
plt.xlabel('Information $I(m; c_0)$ (bits)')
plt.show()

# Conditional probability for final lambda value
# plt.contourf(log(c), m, Pmci, 64, cmap=cmap)
# plt.colorbar()
# plt.ylabel('$P(m|c)$')
# plt.xlabel('Ligand $c$')

# Export plot

exportOptions = {
    'Format': 'eps2',
    'Color': 'rgb',
    'Width': 10,
    'Resolution': 300,
    'FontMode': 'fixed',
    'FontSize': 10,
    'LineWidth': 2
}

# TODO: convert to Python
# ylim([0 0.01])
# xlim([0 2])
# set(gcf, 'PaperUnits', 'centimeters')
# filename = ['informationfitnesscurve'];
# exportfig(gcf, [filename '.eps'], exportOptions);
# system(['epstopdf ' filename '.eps']);
# system(['rm ' filename '.eps']);
