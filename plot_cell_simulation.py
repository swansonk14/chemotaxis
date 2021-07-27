"""Plots the movement of cells in a RapidCell simulation."""
from pathlib import Path

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import Tap


# Header for RapidCell output file
NAMES = ['t', 'x', 'y', 'orientation', 'CheA-P', 'CheY-P', 'methylation', 'CCW_bias']


class Args(Tap):
    data_path: Path  # Path to a file containing the output from a RapidCell simulation.


def plot_cell_simulation(args: Args) -> None:
    """Plots the movement of cells in a RapidCell simulation."""
    # Load data
    data = pd.read_csv(args.data_path, sep='\t', names=NAMES, index_col=False)

    # Extract line segments from cell movement
    points = np.array([data['x'], data['y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Plot cell movements
    fig, ax = plt.subplots()
    lc = LineCollection(segments, cmap='viridis')
    lc.set_array(data['t'])
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    ax.set_xlim(data['x'].min(), data['x'].max())
    ax.set_ylim(data['y'].min(), data['y'].max())
    plt.show()


if __name__ == '__main__':
    plot_cell_simulation(Args().parse_args())
