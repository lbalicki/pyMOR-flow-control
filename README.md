# Model Order Reduction of Linearized NSE with pyMOR

The Jupyter Notebook `pymor_nse_control.ipynb` demonstrates model order reduction
of the linearized Navier-Stokes equation which occurs in the control setup
described [here](https://arxiv.org/pdf/1707.08711.pdf).
In particular a low-order controller is realized using the reduced model, which
allows for efficient stabilization of incompressible flows in the simulations.

Helper functions used for the simulation and storing of results are taken from
[here](https://zenodo.org/record/834940#.XsUZJBaxU5k).

Model order reduction is performed using [pyMOR](https://github.com/pymor/pymor).

The repository contains [results](./data/lvl_2/re_110) for the simulation without
control as well as [results](./data/lvl_2_bc/re_110_palpha_0.001) with boundary control.
Run `paraview v_results.pvd` or `paraview p_results.pvd` in the respective folder
in order to view the results for the velocity or pressure, respectively.

Additionally, the [experiments directory](./experiments) contains scripts and
results for numerical experiments. Run `make` or `python run_experiments.py` in
the directory in order to compute the results.

## Installation
Install instructions are available [here](./INSTALL.md).

## License
The code is available under [MIT License](./LICENSE.txt).
