# TESLA-KIT 

Teslakit is a Python3 library for statistical calculations and methodologies for handling global climate data.
The stochastic climate emulator proposed is built on the recognition that coastal conditions are the result of meteorological forcing, 
and that synoptic-scale meteorology is in turn a consequence of large-scale quasi-steady atmospheric and oceanic patterns (e.g., Hannachi & Legras, 1995)


## Main contents


## Project Map

![picture](docs/img/map.svg)


## Documentation


## Install 
- - -

Source code is currently hosted on Bitbucket at: https://gitlab.com/ripollcab/teslakit/tree/master 

### Installing from sources

Navigate to the base root of [teslakit](./)

Using a Python virtual environment is recommended

```
# install virtualenv package 
python3 -m pip install virtualenv

# create a new virtual environment for teslakit installation
python3 -m virtualenv venv

# now activate the virtual environment
source venv/bin/activate
```

Now install teslakit requirements

```
pip install -r requirements.txt
```

Then install teslakit

```
python setup.py install
```

### Installing optional modules

Basemap module is used in some Teslakit figures.

It is not needed to run any calculation and installing it is optional

Follow Basemap installation instructions at https://github.com/matplotlib/basemap


## Handling a Teslakit Project 
- - -

Jupyter notebook files can be found at [notebooks](notebooks/)

launch jupyter notebook

```
jupyter notebook
```

Current development test site notebooks can be found at [ROI](notebooks/nb_ROI/)


Input data adquisition is currently not integrated in teslakit.

Test site input data list can be found at: [00_Set_Database.ipynb](notebooks/nb_ROI/00_Set_Database.ipynb)


## Contributors


## Thanks also to


## License

This project is licensed under the MIT License - see the [license](LICENSE.txt) file for details




