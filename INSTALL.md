# Installation on Ubuntu 18.04. LTS

Make sure Python 3 is installed on your system.
It is recommended to use a virtual environment for the installation.
Below are the instructions for using virtualenvwrapper, however, creating the environment with the `virtualenv` command will also work.

## Installation using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)
Make sure [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) is installed according to their install instructions.
E.g. run:
```
pip install --user virtualenvwrapper
export WORKON_HOME=~/Envs
mkdir -p $WORKON_HOME
source /usr/local/bin/virtualenvwrapper.sh
```

Create a virtual environment based on Python 3, e.g. by running
```
mkvirtualenv --python=/usr/bin/python3 pyMOR_flow_control
```

Install the pyMOR submodule (which is [this](https://github.com/lbalicki/pymor/tree/StokesDescriptorModel) forked pyMOR repository).
E.g. run:
```
git submodule init
git submodule update
cd pymor/
pip install -e .[full]
```
This will also install other required packages such as jupyter, numpy, etc.

For visualization install [Paraview](https://www.paraview.org/) e.g. via `sudo apt install paraview`.
Or download the recommended version 4.4.0 from [here](https://www.paraview.org/download/) and install from source accordingly.

In order to access the virtual environment in the jupyter notebook, you can add a ipykernel by running the following (from outside the virtual environment):
```
pip install --user ipykernel
python -m ipykernel install --user --name=pyMOR_flow_control
```

The notebook can now be accessed via running `jupyter notebook pymor_nse_control.ipynb`.
Make sure to select the newly created ipykernel in the notebook.
