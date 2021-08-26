"""Plots the movement of cells in a RapidCell simulation."""
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
    DPI,
    DRIFT_UNITS,
    FIGSIZE,
    INFORMATION_UNITS,
    LIGAND_UNITS,
    METHYLATION_MAX,
    METHYLATION_MIN,
    compute_mutual_information,
    plot_output,
    set_plot_parameters,
    set_up_methylation_levels_and_ligand_concentrations
)


# Header for RapidCell output file
COLUMNS = ['x', 'y', 'orientation', 'CheA-P', 'CheY-P', 'methylation', 'CCW_bias']

# Arena boundaries
X_MIN = Y_MIN = 0.0
X_MAX = Y_MAX = 16.0
X_FOR_MAX_METH = 14.4


class Args(Tap):
    data_path: Path
    """Path to a file containing the output from a RapidCell simulation."""
    color_gradients: List[Literal['time', 'orientation', 'CheA-P', 'CheY-P', 'methylation', 'CCW_bias', 'drift']] = ['time']
    """The parameter(s) to use to determine the color gradient."""
    polyfit: bool = False
    """Whether to fit a polynomial to the ligand-methylation data."""
    poly_degree: int = 7
    """Degree of the polynomial to fit to the ligand-methylation data."""
    save_dir: Path = None
    """Path to directory where plots will be saved. If None, plots are displayed instead."""

    def process_args(self) -> None:
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)


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
                    color_gradient: str,
                    save_path: Path = None) -> None:
    """
    Plots the paths of the cells in the simulation.

    :param data: A list of dictionaries containing data for each cell in the simulation.
    :param color_gradient: The parameter to use to determine the color gradient.
    :param save_path: Path where plot will be saved. If None, plot is displayed instead.
    """
    # Plot cell movements
    fig, ax1 = plt.subplots(figsize=FIGSIZE * 1.5)

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
        lc.set_rasterized(True)
        lc.set_array(cell_data[color_gradient])
        lc.set_linewidth(2)
        ax1.add_collection(lc)

        # Plot start and end
        ax1.scatter(cell_data['x'][0], cell_data['y'][0], color='red', marker='o', s=15, zorder=2)
        ax1.scatter(cell_data['x'][-1], cell_data['y'][-1], color='red', marker='x', s=15, zorder=2)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax1)
    cbar.set_label(color_gradient.replace('_', ' '))
    ax1.set_xlim(X_MIN, X_MAX)
    ax1.set_ylim(Y_MIN, Y_MAX)
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')

    # Plot ligand concentration
    ax2 = ax1.twiny()
    ax2.set_xlabel(r'Ligand concentration $\log_{10}(c)$ ' + f'({LIGAND_UNITS})', color='r')
    ax2.tick_params(axis='x', colors='red')
    ax2.spines['top'].set_color('red')
    ax2.set_xlim(log_ligand_concentration(X_MIN), log_ligand_concentration(X_MAX))

    # Average drift
    avg_drift = np.mean([(cell_data['x'][-1] - cell_data['x'][0]) / cell_data['time'][-1] for cell_data in data])
    ax1.scatter([], [], s=3, color='purple', label=f'Average drift = {1000 * avg_drift:.2f} {DRIFT_UNITS}')

    # Average log(c)
    avg_log_c = np.mean([concentration for cell_data in data for concentration in cell_data['ligand']])
    ax1.scatter([], [], s=3, color='purple', label=r'Average $\log_{10}$(c) = ' + f'{avg_log_c:.2f} {LIGAND_UNITS}')

    ax1.legend(loc='upper left')

    if save_path is not None:
        plt.savefig(save_path, dpi=DPI)
    else:
        plt.show()

    plt.close()


def plot_drift(data: List[Dict[str, np.ndarray]],
               num_bins: int = 1000,
               min_count: int = 10,
               save_path: Path = None) -> None:
    """
    Plots the average drift of the cells across different bins.

    :param data: A list of dictionaries containing data for each cell in the simulation.
    :param num_bins: Number of bins to divide the x range into.
    :param min_count: Minimum number of counts in each bin to display its average drift.
    :param save_path: Path where plot will be saved. If None, plot is displayed instead.
    """
    # Set up x range bins
    bins = np.linspace(X_MIN, X_MAX, num_bins + 1)
    bin_size = (X_MAX - X_MIN) / num_bins
    bin_centers = bins[:-1] + bin_size / 2
    drift_sums = np.zeros(num_bins)
    drift_counts = np.zeros(num_bins)

    # Iterate over individual cells
    for cell_data in tqdm(data):
        current_bin = max(0, np.searchsorted(bins, cell_data['x'][0]) - 1)
        start_time = None

        for x, time in zip(cell_data['x'], cell_data['time']):
            bin = max(0, np.searchsorted(bins, x) - 1)

            if bin != current_bin:
                # Don't count first crossing of bin boundary since cell starts in the middle of a bin
                if start_time is not None:
                    if bin == current_bin + 1:
                        drift_sums[bin] += bin_size / (time - start_time)
                    elif bin == current_bin - 1:
                        drift_sums[bin] -= bin_size / (time - start_time)
                    else:
                        raise ValueError('Skipped more than one bin in one time step.')

                    drift_counts[bin] += 1

                current_bin = bin
                start_time = time

    drift_averages = drift_sums / (drift_counts + (drift_counts == 0))  # mask to ensure no divide by zero error
    drift_averages *= 1000  # Convert from mm/s to um/s

    fig, ax = plt.subplots(figsize=FIGSIZE * 1.2)
    ax.scatter(bin_centers[drift_counts >= min_count], drift_averages[drift_counts >= min_count], s=10)
    ax.set_xlabel('X bin')
    ax.set_ylabel(f'Average drift ({DRIFT_UNITS})')
    ax.set_title('Average drift across bins')

    if save_path is not None:
        plt.savefig(save_path, dpi=DPI)
    else:
        plt.show()


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


def poly_str_java(poly: Polynomial,
                  var_name: str = 'logS') -> str:
    """
    Converts a polynomial into a line of Java code.

    :param poly: A Polynomial object.
    :param var_name: The name of the x variable in the polynomial.
    :return: A string representation of Java code to compute the polynomial.
    """
    return 'meth = ' + ''.join(
        f'{"" if i == 0 else " + " if coef >= 0 else " - "}'  # sign
        f'{abs(coef)}'  # coefficient
        f'{"" if i == 0 else f" * Math.pow({var_name}, {i})"}'  # x power
        for i, coef in enumerate(poly.coef)
    ) + ';'


def compute_ligand_methylation_counts(data: List[Dict[str, np.ndarray]],
                                      log_c: np.ndarray,
                                      m: np.ndarray) -> np.ndarray:
    """
    Computes the count of ligand-methylation pairs.

    :param data: A list of dictionaries containing data for each cell in the simulation.
    :param log_c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :return: A matrix containing the counts of ligand-methylation pairs.
    """
    # Set up count grid
    mi, log_ci = m[:, 0], log_c[0]
    m_c_count_grid = np.zeros(m.shape)

    # Count occurrences of ligand-methylation pairs
    for cell_data in tqdm(data):
        c_indices = np.searchsorted(log_ci, cell_data['ligand'])
        m_indices = np.searchsorted(mi, cell_data['methylation'])

        for m_index, c_index in zip(m_indices, c_indices):
            m_c_count_grid[m_index, c_index] += 1

    return m_c_count_grid


def compute_empirical_pmc(m_c_count_grid: np.ndarray) -> np.ndarray:
    """
    Computes the empirical conditional distribution P(m|c) of the methylation level given the ligand concentration.

    :param m_c_count_grid: A matrix containing the counts of ligand-methylation pairs.
    :return: A numpy array containing the empirical conditional distribution P(m|c).
    """
    # Normalize ligand-methylation counts
    m_c_count_norm = m_c_count_grid.sum(axis=0)
    Pmc = m_c_count_grid / (m_c_count_norm + (m_c_count_norm == 0))  # mask to ensure no divide by zero error

    return Pmc


def fit_polynomial(m_c_count_grid: np.ndarray,
                   log_c: np.ndarray,
                   m: np.ndarray,
                   poly_degree: int,
                   verbose: bool = False) -> Polynomial:
    """
    Fits a polynomial to the empirical distribution of methylation levels and ligand concentrations.

    :param m_c_count_grid: A matrix containing the counts of ligand-methylation pairs.
    :param log_c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param poly_degree: The degree of the polynomial to fit to the ligand-methylation data.
    :param verbose: Whether to print additional information like the polynomial in Java syntax.
    :return: The Polynomial that has been fit to the ligand-methylation data.
    """
    X, Y = [], []
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] < 8.0:
                X += [log_c[i, j]] * int(m_c_count_grid[i, j])
                Y += [m[i, j]] * int(m_c_count_grid[i, j])

    polynomial = Polynomial.fit(X, Y, deg=poly_degree, window=[min(X), max(X)])

    if verbose:
        print(poly_str_java(polynomial))

    return polynomial


def plot_ligand_methylation_distribution(Pmc: np.ndarray,
                                         log_c: np.ndarray,
                                         m: np.ndarray,
                                         info: float = None,
                                         polynomial: Polynomial = None,
                                         save_path: Path = None) -> None:
    """
    Plots the distribution of methylation levels given ligand concentrations.

    :param Pmc: The conditional distribution P(m|c) of the methylation level given the ligand concentration.
    :param log_c: A matrix of ligand concentrations (differing across the columns).
    :param m: A matrix of methylation levels (differing across the rows).
    :param info: Mutual information of Pmc.
    :param polynomial: A Polynomial that has been fit to the P(m|c) data.
    :param save_path: Path where plot will be saved. If None, plot is displayed instead.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE * 1.2)

    if polynomial is not None:
        log_ci = log_c[0]
        y_fit = polynomial(log_ci)
        indices_over_8 = np.where(y_fit > 8.0)[0]
        first_index_before_8 = max(0, indices_over_8[0] - 1) if len(indices_over_8) > 0 else None
        ax.plot(log_ci[:first_index_before_8], y_fit[:first_index_before_8], color='red', label=poly_str(polynomial))

    ax.scatter([], [], s=3, color='cyan', label=f'Mutual information = {info:.2f} {INFORMATION_UNITS}')
    im = ax.contourf(log_c, m, Pmc, levels=64, cmap=plt.get_cmap('viridis'), zorder=-20)
    ax.set_rasterization_zorder(-10)
    fig.colorbar(im, ax=ax)
    ax.legend(loc='upper left')
    ax.set_title('$P(m|c)$')
    ax.set_xlabel(r'Ligand concentration $\log_{10}(c)$ ' + f'({LIGAND_UNITS})')
    ax.set_ylabel('Methylation level $m$')

    if save_path is not None:
        plt.savefig(save_path, dpi=DPI)
    else:
        plt.show()


def plot_cell_simulation(args: Args) -> None:
    """Plots the movement of cells in a RapidCell simulation."""
    # Set plot parameters
    set_plot_parameters()

    # Save arguments
    if args.save_dir is not None:
        args.save(args.save_dir / 'args.json')

    # Load data
    data = load_data(path=args.data_path)

    # Compute drift and ligand concentrations
    for cell_data in tqdm(data):
        cell_data['drift'] = compute_running_average_drift(x=cell_data['x'], time=cell_data['time'])
        cell_data['ligand'] = log_ligand_concentration(cell_data['x'])

    # Plot cell paths
    for color_gradient in args.color_gradients:
        plot_cell_paths(
            data=data,
            color_gradient=color_gradient,
            save_path=args.save_dir / f'cell_paths_{color_gradient}.pdf' if args.save_dir is not None else None
        )

    # Plot drift
    plot_drift(
        data=data,
        save_path=args.save_dir / 'drift.pdf' if args.save_dir is not None else None
    )

    # Set up methylation levels m and ligand concentrations
    m, c = set_up_methylation_levels_and_ligand_concentrations(
        m_min=METHYLATION_MIN,
        m_max=METHYLATION_MAX,
        log_c_min=log_ligand_concentration(X_MIN),
        log_c_max=log_ligand_concentration(X_MAX)
    )
    log_c = np.log10(c)

    # Count empirical ligand-methylation pairs
    m_c_count_grid = compute_ligand_methylation_counts(data=data, log_c=log_c, m=m)

    # Compute empirical conditional distribution P(m|c) by normalizing ligand-methylation counts
    Pmc = compute_empirical_pmc(m_c_count_grid=m_c_count_grid)

    # Compute empirical marginal distribution P(c)
    Pc = np.broadcast_to((m_c_count_grid.sum(axis=0) / m_c_count_grid.sum())[np.newaxis, :], Pmc.shape)

    # Compute empirical marginal distribution P(m)
    Pm = np.broadcast_to((m_c_count_grid.sum(axis=1) / m_c_count_grid.sum())[:, np.newaxis], Pmc.shape)

    # Compute mutual information
    info = compute_mutual_information(Pmc=Pmc, Pc=Pc, Pm=Pm, c=c, m=m)

    # Optionally fit a polynomial to P(m|c)
    if args.polyfit:
        polynomial = fit_polynomial(m_c_count_grid=m_c_count_grid, log_c=log_c, m=m, poly_degree=args.poly_degree)
    else:
        polynomial = None

    # Plot P(m|c) including optional polynomial fit
    plot_ligand_methylation_distribution(
        Pmc=Pmc,
        log_c=log_c,
        m=m,
        info=info,
        polynomial=polynomial,
        save_path=args.save_dir / 'pmc.pdf' if args.save_dir is not None else None
    )

    # Plot P(c)
    plot_output(
        output=Pc,
        title='$P(c)$',
        c=c,
        m=m,
        save_path=args.save_dir / 'pc.pdf' if args.save_dir is not None else None
    )

    # Plot P(m)
    plot_output(
        output=Pm,
        title='$P(m)$',
        c=c,
        m=m,
        save_path=args.save_dir / 'pm.pdf' if args.save_dir is not None else None
    )


if __name__ == '__main__':
    plot_cell_simulation(Args().parse_args())
