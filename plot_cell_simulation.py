"""Plots the movement of cells in a RapidCell simulation."""
from pathlib import Path
from typing import Literal

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from tap import Tap
from tqdm import tqdm


# Header for RapidCell output file
COLUMNS = ['x', 'y', 'orientation', 'CheA-P', 'CheY-P', 'methylation', 'CCW_bias']


class Args(Tap):
    data_path: Path
    """Path to a file containing the output from a RapidCell simulation."""
    color_gradient: Literal['time', 'orientation', 'CheA-P', 'CheY-P', 'methylation', 'CCW_bias', 'drift'] = 'time'
    """The parameter to use to determine the color gradient."""


def log_ligand_concentration(x: float, rate: float = 0.001) -> float:
    """
    Computes the log ligand concentration using and exponential gradient.

    :param x: The x position.
    :param rate: The exponential rate.
    :return: The log of the ligand concentration.
    """
    return np.log(rate * np.exp(x))


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


def plot_cell_simulation(args: Args) -> None:
    """Plots the movement of cells in a RapidCell simulation."""
    # Load data
    data = np.loadtxt(args.data_path)

    # Map from cell index to array containing data for that cell
    data = [{
        'time': data[:, 0],
        **{column: data[:, i + j].T for j, column in enumerate(COLUMNS)}
    } for i in range(1, data.shape[1], len(COLUMNS))]

    # Plot cell movements
    fig, ax1 = plt.subplots()

    # Iterate over individual cells
    for cell_data in tqdm(data):
        # Running average drift
        cell_data['drift'] = compute_running_average_drift(x=cell_data['x'], time=cell_data['time'])

        # Extract line segments from cell movement
        points = np.array([cell_data['x'], cell_data['y']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Plot line segments
        lc = LineCollection(segments, cmap='viridis')
        lc.set_array(cell_data[args.color_gradient])
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
    cbar.set_label(args.color_gradient)
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


if __name__ == '__main__':
    plot_cell_simulation(Args().parse_args())
