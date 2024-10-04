# Distributed Data Fusion for Space Based Target Tracking

## Overview
Simulates cooperative localization in discrete time using satellite agents sensing targets near earth. Each
agent individually estimates the position and velocity of targets in FOV and communicates infromation with
their neighbors for collective estimation.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
4. [Contact](#contact)

## Installation

### Linux
```bash
source ./setup.sh
```

### Windows
```bash
## Begin by installing python 3.10:
https://www.python.org/downloads/release/python-3100/

# Create a python virtual environment from a terminal, on windows use powershell.
py -3.10 -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate

## Install dependencies
pip install -r requirements_lock.txt
```

## Usage
```bash
# Different yaml files are specifies for different mission scenarios in "scenarios/"

# Run the simulation
# can also use the `satellite_ddf` entrypoint
python phase3/main.py

# Estimation Results
State Estimation Error: "plots"

# Gaussian Uncertainty Ellipsoids
Gasussian Uncertainty Ellipsoids: "gifs"

# Simulation Visualization
Simulation Visualization: "gifs"

# Data Dump File
Data Dump File: "data"

# Modify the simulation parameters in scenarios/default.yaml file
#   - Number of satellites
#   - Number of targets
#   - Placement of satellites and targets
#   - Sensor parameters
#   - Communication Strucutre
```

## Contact
```
Nolan Stevenson - nolan.stevenson@colorado.edu
Aidan Bagley - aidan.bagley@colorado.edu
Ryan Draves - ryan.draves@colorado.edu
```
