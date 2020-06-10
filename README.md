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
pip install -r requirements/requirements.txt
```

Then install teslakit

```
python setup.py install
```

### Installing SWAN numerical model


[teslakit/numerical\_models/swan/](./teslakit/numerical_models/swan/) is a custom developed python toolbox used to wrap SWAN numerical model

SWAN numerical model has to be compiled for serial execution and stored at teslakit swan binary [resources](./teslakit/numerical_models/swan/resources/) folder

First download and compile SWAN serial executable

```
  # you may need to install a fortran compiler
  sudo apt install gfortran

  # download and unpack
  wget http://swanmodel.sourceforge.net/download/zip/swan4131.tar.gz
  tar -zxvf swan4131.tar.gz

  # compile numerical model
  cd swan4131/
  make config
  make ser
```

Now manually move executable file to [resources/swan\_bin/](./teslakit/numerical_models/swan/resources/swan_bin/) and change file name to "swan\_ser.exe"

Alternatively, set swan executable file using our swan python module

```
  # Launch a python interpreter
  $ python

  Python 3.6.9 (default, Apr 18 2020, 01:56:04) 
  [GCC 8.4.0] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  
  >>> from teslakit.numerical_models import swan
  >>> swan.set_swan_binary_file('swan.exe')
```


### Installing optional modules

Basemap module is used in some Teslakit figures.

It is not needed to run any calculation and installing it is optional

Follow Basemap installation instructions at https://github.com/matplotlib/basemap

```
pip install git+https://github.com/matplotlib/basemap.git
```

## Handling a Teslakit Project 
- - -

Jupyter notebook files can be found at [notebooks](notebooks/)

launch jupyter notebook

```
jupyter notebook
```

Current development test site notebooks can be found at [ROI](notebooks/nb_ROI/)

Also, test site needed input files can be downloaded from [OneDrive](https://unican-my.sharepoint.com/:u:/g/personal/ripolln_unican_es/EbV8pLPheXZEpg8I6VCKjF4BscJoReoq5pO5w0358x88Vg?e=ygozCh)

(Input data adquisition is currently not integrated in teslakit)


Once ROI data is downloaded and unpacked, a input data check can be done at: [00_Set_Database.ipynb](notebooks/ROI/01_Offshore/00_Set_Database.ipynb)


## Contributors


## Thanks also to


## License

This project is licensed under the MIT License - see the [license](LICENSE.txt) file for details




