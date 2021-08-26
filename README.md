# E. Coli Chemotaxis and Mutual Information

This repository contains code for evaluating the trade-offs between drift, entropy production, and mutual information in *E. coli* chemotaxis.

## Installation

```
conda env create -f environment.yml
```

## Setup

```
conda activate chemotaxis
```

## Usage

To compute the trade-offs between drift, entropy production, and mutual information, run the following command. The full list of command line options can be seen in the `Args` class in `rate_distortion_drift.py`.
```
python rate_distortion_drift.py
```

To visualize the behaviour of cells that employ the optimal conditional distributions computed above, first run this fork of the cell simulator RapidCell: https://github.com/swansonk14/rapidcell. When running the simulation, change the ligand-methylation relation in the Network tab as desired to simulate different optimal conditional distributions. Then, run the following command. The full list of command line options can be seen in the `Args` class in `plot_cell_simulation.py`.
```
python plot_cell_simulation.py --data_path /path/to/rapidcell/individuals.txt
```
