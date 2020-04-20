# Installation on Ubuntu 18.04. LTS

Clone or download [this](https://github.com/lbalicki/pymor/tree/StokesDescriptorModel) forked pyMOR repository and install it.
E.g. run:
```
git clone https://github.com/lbalicki/pymor
cd $PYMOR_SOURCE_DIR
git checkout StokesDescriptorModel
pip install -e .[full]
```

For visualization install [Paraview](https://www.paraview.org/) e.g. via `sudo apt install paraview`.

The notebook can now be accessed via running `jupyter notebook pymor_nse_control.ipynb`.
