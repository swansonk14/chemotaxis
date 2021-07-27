"""Plots the movement of cells in a RapidCell simulation."""
from pathlib import Path
from typing import Dict, List, Literal

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from tap import Tap


# Header for RapidCell output file
COLUMNS = ['x', 'y', 'orientation', 'CheA-P', 'CheY-P', 'methylation', 'CCW_bias']


class Args(Tap):
    data_path: Path
    """Path to a file containing the output from a RapidCell simulation."""
    color_gradient: Literal['time', 'orientation', 'CheA-P', 'CheY-P', 'methylation', 'CCW_bias'] = 'time'
    """The parameter to use to determine the color gradient."""


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
    fig, ax = plt.subplots()

    # Iterate over individual cells
    for cell_data in data:
        # Extract line segments from cell movement
        points = np.array([cell_data['x'], cell_data['y']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Plot line segments
        lc = LineCollection(segments, cmap='viridis')
        lc.set_array(cell_data[args.color_gradient])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

    X = [x for cell_data in data for x in cell_data['x']]
    Y = [y for cell_data in data for y in cell_data['y']]

    fig.colorbar(line, ax=ax)
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_ylim(np.min(Y), np.max(Y))
    plt.show()


if __name__ == '__main__':
    plot_cell_simulation(Args().parse_args())
