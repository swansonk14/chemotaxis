"""Plots the movement of cells in a RapidCell simulation."""
from itertools import product
from pathlib import Path
from typing import Dict, List, Literal, Union

from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from tap import Tap
from tqdm import tqdm, trange

from rate_distortion_drift import (
    CMAP,
    METHYLATION_MAX,
    METHYLATION_MIN,
    set_up_methylation_levels_and_ligand_concentrations
)


# Header for RapidCell output file
COLUMNS = ['x', 'y', 'orientation', 'CheA-P', 'CheY-P', 'methylation', 'CCW_bias']

# Arena boundaries
X_MIN = Y_MIN = 0.0
X_MAX = Y_MAX = 20.0


class Args(Tap):
    data_path: Path
    """Path to a file containing the output from a RapidCell simulation."""
    color_gradient: Literal['time', 'orientation', 'CheA-P', 'CheY-P', 'methylation', 'CCW_bias', 'drift'] = 'time'
    """The parameter to use to determine the color gradient."""
    polyfit: bool = False
    """Whether to fit a polynomial to the ligand-methylation data."""


def log_ligand_concentration(x: Union[float, np.ndarray], rate: float = 0.001) -> Union[float, np.ndarray]:
    """
    Computes the log (base 10) ligand concentration using and exponential gradient.

    :param x: The x position(s).
    :param rate: The exponential rate.
    :return: The log (base 10) of the ligand concentration(s).
    """
    return np.log10(rate * np.exp(x))


def compute_running_average_drift(x: np.ndarray,
                                  time: np.ndarray,
                                  window_size: int = 11) -> np.ndarray:
    """
    Computes the running average drift.

    :param x: X locations.
    :param time: Time points.
    :param window_size: The window size to use when computing average drift.
    :return: An array of running average drift velocities in um/s.
    """
    running_average_drift = np.zeros(len(x))

    for i in range(1, len(x)):
        prev = max(0, i - window_size)
        running_average_drift[i] = 1000 * (x[i] - x[prev]) / (time[i] - time[prev])

    return running_average_drift


def load_data(path: Path) -> List[Dict[str, np.ndarray]]:
    """
    Loads cell data from a RapidCell simulation.

    :param path: Path to an output file from a RapidCell simulation.
    :return: A list of dictionaries containing data for each cell in the simulation.
    """
    # Load data
    data = np.loadtxt(path)

    # Map from cell index to array containing data for that cell
    data = [{
        'time': data[:, 0],
        **{column: data[:, i + j].T for j, column in enumerate(COLUMNS)}
    } for i in trange(1, data.shape[1], len(COLUMNS))]

    return data


def plot_cell_paths(data: List[Dict[str, np.ndarray]],
                    color_gradient: str) -> None:
    """
    Plots the paths of the cells in the simulation.

    :param data: A list of dictionaries containing data for each cell in the simulation.
    :param color_gradient: The parameter to use to determine the color gradient.
    """
    # Plot cell movements
    fig, ax1 = plt.subplots()

    # Get min and max for color gradient normalization
    color_data = [c for cell_data in data for c in cell_data[color_gradient]]
    norm = Normalize(vmin=np.min(color_data), vmax=np.max(color_data))

    # Iterate over individual cells
    for cell_data in tqdm(data):
        # Extract line segments from cell movement
        points = np.array([cell_data['x'], cell_data['y']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Plot line segments
        lc = LineCollection(segments, norm=norm, cmap=CMAP)
        lc.set_array(cell_data[color_gradient])
        lc.set_linewidth(2)
        ax1.add_collection(lc)

        # Add drift at final location
        drift = (cell_data['x'][-1] - cell_data['x'][0]) / (cell_data['time'][-1] - cell_data['time'][0])
        ax1.text(cell_data['x'][-1], cell_data['y'][-1], rf'{1000 * drift:.2f} $\mu$m/s', color='b')

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax1)
    cbar.set_label(color_gradient)
    ax1.set_xlim(X_MIN, X_MAX)
    ax1.set_ylim(Y_MIN, Y_MAX)
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')

    # Plot ligand concentration
    ax2 = ax1.twiny()
    ax2.set_xlabel(r'Ligand concentration $\log(c)$', color='r')
    ax2.tick_params(axis='x', colors='red')
    ax2.spines['top'].set_color('red')
    ax2.set_xlim(log_ligand_concentration(X_MIN), log_ligand_concentration(X_MAX))

    plt.show()
    plt.close()


def poly_str(poly: Polynomial) -> str:
    """
    Converts a polynomial into a LaTeX string.

    :param poly: A Polynomial object.
    :return: A LaTeX string representation of the Polynomial.
    """
    return ''.join(
        f'{"" if i == 0 else "$+$" if coef >= 0 else "$-$"}'  # sign
        f'{abs(coef):.2e}'  # coefficient
        f'{"" if i == 0 else f"$x^{i}$"}'  # x power
        for i, coef in enumerate(poly.coef)
    )


def plot_ligand_methylation_distribution(data: List[Dict[str, np.ndarray]], polyfit: bool) -> None:
    """
    Plots the distribution of methylation levels given ligand concentrations.

    :param data: A list of dictionaries containing data for each cell in the simulation.
    :param polyfit: Whether to fit a polynomial to the data.
    """
    # Set up ligand and methylation grid
    ligand_concentrations = [l for cell_data in data for l in cell_data['ligand']]
    methylation_levels = [m for cell_data in data for m in cell_data['methylation']]
    log_c_min, log_c_max = np.min(ligand_concentrations), np.max(ligand_concentrations)
    m_min, m_max = min(METHYLATION_MIN, np.min(methylation_levels)), max(METHYLATION_MAX, np.max(methylation_levels))

    m, c = set_up_methylation_levels_and_ligand_concentrations(
        m_min=m_min,
        m_max=m_max,
        log_c_min=log_c_min,
        log_c_max=log_c_max
    )
    log_c = np.log10(c)
    mi, log_ci = m[:, 0], log_c[0]
    count_grid = np.zeros(m.shape)

    # Count occurances of ligand-methylation pairs
    for cell_data in tqdm(data):
        valid_indices = (log_c_min <= cell_data['ligand']) & \
                        (cell_data['ligand'] <= log_c_max) & \
                        (m_min <= cell_data['methylation']) & \
                        (cell_data['methylation'] <= m_max)

        ligand = cell_data['ligand'][valid_indices]
        methylation = cell_data['methylation'][valid_indices]

        c_indices = np.searchsorted(log_ci, ligand)
        m_indices = np.searchsorted(mi, methylation)

        for m_index, c_index in zip(m_indices, c_indices):
            count_grid[m_index, c_index] += 1

    # Normalize ligand-methylation counts
    count_norm = count_grid.sum(axis=0)
    Pmc = count_grid / (count_norm + (count_norm == 0))  # mask to ensure no divide by zero error

    plt.contourf(log_c, m, Pmc, levels=64, cmap=plt.get_cmap('viridis'))
    plt.colorbar()

    # Fit a polynomial to the data
    if polyfit:
        X, Y = [], []
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j] < 8.0:
                    X += [log_c[i, j]] * int(count_grid[i, j])
                    Y += [m[i, j]] * int(count_grid[i, j])

        poly = Polynomial.fit(X, Y, deg=7, window=[min(X), max(X)])

        y_fit = poly(log_ci)
        indices_over_8 = np.where(y_fit > 8.0)[0]
        first_index_before_8 = max(0, indices_over_8[0] - 1) if len(indices_over_8) > 0 else None

        plt.plot(log_ci[:first_index_before_8], y_fit[:first_index_before_8], color='red', label=poly_str(poly))
        plt.legend()

    plt.title('$P(m|c)$')
    plt.xlabel(r'Ligand concentration $\log_{10}(c)$')
    plt.ylabel('Methylation level $m$')
    plt.show()


def plot_cell_simulation(args: Args) -> None:
    """Plots the movement of cells in a RapidCell simulation."""
    # Load data
    data = load_data(path=args.data_path)

    # Compute drift and ligand concentrations
    for cell_data in tqdm(data):
        cell_data['drift'] = compute_running_average_drift(x=cell_data['x'], time=cell_data['time'])
        cell_data['ligand'] = log_ligand_concentration(cell_data['x'])

    # Plot cell paths
    plot_cell_paths(data=data, color_gradient=args.color_gradient)

    # Plot distribution of methylation levels given ligand concentrations
    plot_ligand_methylation_distribution(data=data, polyfit=args.polyfit)


if __name__ == '__main__':
    plot_cell_simulation(Args().parse_args())
