# Distributed Data Fusion for Space Based Target Tracking 

## Overview
Simulates cooperative localization in discrete time using satellite agents sensing targets near earth. Each
agent individually estimates the position and velocity of targets in FOV and communicates infromation with 
their neighbors for collective estimation. 

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Contact](#contact)

## Installation
```bash
# Begin by installing python 3.10, is the version of python that used:
# for windowsOS:
https://www.python.org/downloads/release/python-3100/
# for linux:
apt install python3.10

# Create a python virtual environment in a python terminal
py -3.10 -m venv env
# Note, you can change "env" to whatever you want your virtual environment to be called

# Now go into your virtual environment
# In a terminal:
cd env/

# Activate the virtual environment
# Windows:
Type "activate"
# If you ever want to deactivate type "deactivate"
# Linux:
source bin/activate

# The virtual environment should now be activate.
# In the virtual environment:
# Check that the version of python is correct:
python --version
# It should say Python 3.10
# Now check your virtual environment packages:
pip list
# It should only display pip and setuptools

# Navigate to the include folder of the virtual environment
cd ..\Include\

# Clone the repository
git clone https://github.com/nolans12/Satellite-DDF.git

# Navigate into the repository
cd .\Satellite-DDF\

# Install dependencies
pip install -r requirements.txt
# This may take a while

# Now you should be able to run main.py in phase1 to run the code!
# Use your favorite IDE, such as VSCode, to run the python code.
```

## Usage
```bash
# Run the simulation
python main.py

# Estimation Results
State Estimation Error: "plots"

# Gaussian Uncertainty Ellipsoids
Gasussian Uncertainty Ellipsoids: "gifs"

# Simulation Visualization
Simulation Visualization: "gifs"

# Data Dump File
Data Dump File: "data"

# Modify the simulation parameters in main.py file
#   - Number of satellites
#   - Number of targets
#   - Placement of satellites and targets
#   - Sensor parameters
#   - Communication Strucutre
```
## Project Structure
```
Satellite-DDF
│   README.md
│   requirements.txt
│   .gitignore
│   phase1
│   ├── main.py
│   ├── satelliteClass.py
│   ├── targetClass.py
│   ├── sensorClass.py
│   ├── estimatorClass.py
│   ├── communicationClass.py
│   ├── environmentClass.py
│   ├── import_libraries.py

```

## Contact
```
Nolan Stevenson - nolan.stevenson@colorado.edu
Aidan Bagley - aidan.bagley@colorado.edu
```
