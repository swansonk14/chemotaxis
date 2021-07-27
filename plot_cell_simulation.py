"""Plots the movement of cells in a RapidCell simulation."""
from pathlib import Path

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from tap import Tap


# Header for RapidCell output file
COLUMNS = ['t', 'x', 'y', 'orientation', 'CheA-P', 'CheY-P', 'methylation', 'CCW_bias']


class Args(Tap):
    data_path: Path  # Path to a file containing the output from a RapidCell simulation.


def plot_cell_simulation(args: Args) -> None:
    """Plots the movement of cells in a RapidCell simulation."""
    # Load data
    data = np.loadtxt(args.data_path)
    t = data[:, 0]  # (num_time_points,)
    X = data[:, 1::7].T  # (num_cells, num_time_points)
    Y = data[:, 2::7].T  # (num_cells, num_time_points)

    # Plot cell movements
    fig, ax = plt.subplots()

    # Iterate over individual cells
    for x, y in zip(X, Y):
        # Extract line segments from cell movement
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Plot line segments
        lc = LineCollection(segments, cmap='viridis')
        lc.set_array(t)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

    fig.colorbar(line, ax=ax)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    plt.show()


if __name__ == '__main__':
    plot_cell_simulation(Args().parse_args())
