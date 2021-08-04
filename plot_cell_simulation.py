"""Plots the movement of cells in a RapidCell simulation."""
from pathlib import Path
from typing import Dict, List, Literal, Union

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from tap import Tap
from tqdm import tqdm, trange

from rate_distortion_drift import (
    METHYLATION_MAX,
    METHYLATION_MIN,
    plot_output,
    set_up_methylation_levels_and_ligand_concentrations
)


# Header for RapidCell output file
COLUMNS = ['x', 'y', 'orientation', 'CheA-P', 'CheY-P', 'methylation', 'CCW_bias']


class Args(Tap):
    data_path: Path
    """Path to a file containing the output from a RapidCell simulation."""
    color_gradient: Literal['time', 'orientation', 'CheA-P', 'CheY-P', 'methylation', 'CCW_bias', 'drift'] = 'time'
    """The parameter to use to determine the color gradient."""


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

    # Iterate over individual cells
    for cell_data in tqdm(data):
        # Extract line segments from cell movement
        points = np.array([cell_data['x'], cell_data['y']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Plot line segments
        lc = LineCollection(segments, cmap='viridis')
        lc.set_array(cell_data[color_gradient])
        lc.set_linewidth(2)
        line = ax1.add_collection(lc)

        # Add drift at final location
        drift = (cell_data['x'][-1] - cell_data['x'][0]) / (cell_data['time'][-1] - cell_data['time'][0])
        ax1.text(cell_data['x'][-1], cell_data['y'][-1], rf'{1000 * drift:.2f} $\mu$m/s', color='b')

    X = [x for cell_data in data for x in cell_data['x']]
    Y = [y for cell_data in data for y in cell_data['y']]

    min_x, max_x = np.min(X), np.max(X)
    min_y, max_y = np.min(Y), np.max(Y)

    cbar = fig.colorbar(line, ax=ax1)
    cbar.set_label(color_gradient)
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')

    # Plot ligand concentration
    ax2 = ax1.twiny()
    ax2.set_xlabel(r'Ligand concentration $\log(c)$', color='r')
    ax2.tick_params(axis='x', colors='red')
    ax2.spines['top'].set_color('red')
    ax2.set_xlim(log_ligand_concentration(min_x), log_ligand_concentration(max_x))

    plt.show()
    plt.close()


def plot_ligand_methylation_distribution(data: List[Dict[str, np.ndarray]]) -> None:
    """
    Plots the distribution of methylation levels given ligand concentrations.

    :param data: A list of dictionaries containing data for each cell in the simulation.
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
    drift_grid = np.zeros(m.shape)

    # Count occurances of ligand-methylation pairs
    for cell_data in tqdm(data):
        valid_indices = (log_c_min <= cell_data['ligand']) & \
                        (cell_data['ligand'] <= log_c_max) & \
                        (m_min <= cell_data['methylation']) & \
                        (cell_data['methylation'] <= m_max)

        ligand = cell_data['ligand'][valid_indices]
        methylation = cell_data['methylation'][valid_indices]
        drift = cell_data['drift'][valid_indices]

        c_indices = np.searchsorted(log_ci, ligand)
        m_indices = np.searchsorted(mi, methylation)

        for m_index, c_index, d in zip(m_indices, c_indices, drift):
            count_grid[m_index, c_index] += 1
            drift_grid[m_index, c_index] += d

    # Normalize ligand-methylation counts
    count_norm = count_grid.sum(axis=0)

    Pmc = count_grid / (count_norm + (count_norm == 0))  # ensure no divide by zero error
    avg_drift_grid = drift_grid / (count_grid + (count_grid == 0))  # ensure no divide by zero error

    # Plot ligand-methylation distribution
    plot_output(output=Pmc, title='$P(m|c)$', c=c, m=m, plot_max=False)
    plot_output(output=avg_drift_grid, title='Drift', c=c, m=m, plot_max=False)


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
    plot_ligand_methylation_distribution(data=data)


if __name__ == '__main__':
    plot_cell_simulation(Args().parse_args())
