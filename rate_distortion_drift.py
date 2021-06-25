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

# Variables
L = 1000  # Number of intervals
mi = np.linspace(0, 8, L)  # Methylation levels
ci = np.logspace(-3, 3, L)  # Ligand levels(LOG SPACE)
dm = np.mean(np.diff(mi))  # Differences between methylation levels
dc = np.mean(np.diff(ci))  # Differences between ligand levels(note: intervals in c are not constant as defined in log space!)

[c, m] = np.meshgrid(ci, np.flip(mi))  # Mesh grid of ligand and methylation levels

# Fitness function - estimated

# Parameters - Micali table S1
N = 5  # Cooperative receptor number(paper uses 13, range[5; 13])
va = 1 / 3  # Fraction of Tar receptors(paper uses 1 / 3)
vs = 2 / 3  # Fraction of Tsr receptors(paper uses 1 / 3)
Kaon = 0.5  # Active receptors dissociation constant Tar(mM, paper uses 1.0)
Kson = 100000  # Active receptors dissociation constant Tsr(mM, paper uses 1E6)
Kaoff = 0.02  # Inactive receptors dissociation constant Tar(mM, paper uses 0.03)
Ksoff = 100  # Inactive receptors dissociation constant Tsr(mM, paper uses 100)
k = 1

nu = 0.01
YT = 9.7  # muM
mT = 1e4
kyp = 100  # muM ^ {-1} s ^ {-1}
kym = nu * kyp
kzp = 30  # s ^ {-1}
kzm = nu * kzp
kRp = 0.0069  # s ^ {-1}
kRm = nu * kRp
kBp = 0.12  # s ^ {-1}
kBm = nu * kBp

rel_grad = 0.75

# Methylation free energy (Monod-Wyman-Changeux (MWC) model, Clausznitzer equation 5)
mE = 1 - 0.5 * m

# Free energy difference of on/off states of receptor complex (Monod-Wyman-Changeux (MWC) model, Clausznitzer equation 5)
dF = N * (
        mE
        + va * np.log((1 + c / Kaoff) / (1 + c / Kaon))
        + vs * np.log((1 + c / Ksoff) / (1 + c / Kson))
)

# Receptor activity (Monod-Wyman-Changeux (MWC) model, Clausznitzer equation 5)
A = 1 / (1 + np.exp(dF))

# Drift velocity
vd = k * A * (1 - A) * N * (va * (c / (c + Kaoff) - c / (c + Kaon)) + vs * (c / (c + Ksoff) - c / (c + Kson))) * rel_grad

# Entropy production from phosphorylation and methylation
Yp = (kyp * A + kzm) * YT / ((kyp + kym) * A + kzp + kzm)
dSdty = (kyp * (YT - Yp) - kym * Yp) * A * np.log((kyp * (YT - Yp)) / (kym * Yp)) + (kzp * Yp - kzm * (YT - Yp)) * np.log((kzp * Yp) / (kzm * (YT - Yp)))
dSdtm = (kRp - kRm) * (1 - A) * mT * np.log(kRp / kRm) + (kBp - kBm) * A**3 * mT * np.log(kBp / kBm)
EP = (dSdty + dSdtm) / 1000  # Make values smaller to work

# Choose an output(drift vd or entropy production EP)
# output = EP
output = vd

# Input - output curve of methylation and ligand concentration vs output
"""
plt.contourf(np.log(c), m, output, 64, cmap=cmap)
plt.colorbar()
maxi = np.argmax(output, axis=0)  # Index in each column corresponding to maximum
plt.plot(np.log(ci), m[maxi, :], color='red')
plt.xlabel('Ligand $c_0$')
plt.ylabel('Methylation $m$')
plt.show()
"""

# Distribution of external states of c
r = 0.1  # 2;
Pc = np.exp(-r * c) / np.sum(np.exp(-r * c) * dc, axis=0)  # TODO: is this supposed to be nearly all 0.000999???

# Functions to iterate


def Eqn2(Pmc):
    return np.sum(Pmc * Pc * dc, axis=1)  # This should give a vector of Nm by 1


def Eqn5(Pm, lam):
    return np.exp(lam * vd) * Pm / np.sum(Pm * np.exp(lam * vd) * dm, axis=0)  # This should give a matrix of Nm by Nc


# Iterate mutual information algorithm

# TODO: why are the numbers slightly different from Matlab? Is it just differences in numerical precision?

iimax = int(2e4)  # 10
etol = 1e-1  # -4, -5
for lam in np.logspace(0, 1, 10):  # (1, 2, 10)
    print(f'Lambda = {lam:.2f}')

    # Pick an initial guess for Pmc and Pm
    Pmi = np.ones(m.shape)
    # print(dm)
    # print(np.sum(Pmi * dm, axis=1)
    Pmi = Pmi / np.sum(Pmi * dm, axis=0)
    # 'this is normalised!'
    Pmci = Eqn5(Pmi, lam)  # columns not normalised!!!!

    for ii in range(iimax):
        # print(ii)
        Pmo = Pmi
        Pmi = Eqn2(Pmci)  # NaN!!!!
        Pmci = Eqn5(Pmi, lam)
        # print(np.linalg.norm(Pmi - Pmo) * dm)

        if np.linalg.norm(Pmi - Pmo) * dm <= etol:
            print(f'Converged for lambda = {lam:.2f} after {ii + 1} iterations')
            Imin = np.sum(dc * Pc * np.sum(dm * Pmci * np.log2(eps + Pmci / (eps + Pmi)), axis=0), axis=1)  # Mutual information Eq.1
            outmax = np.sum(dc * Pc * np.sum(dm * Pmci * output, axis=0), axis=1)  # Mean fitness from Eq.4
            print(outmax[0])
            plt.plot(Imin, outmax, 'x')
            break

breakpoint()

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
