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
## Create a folder for the project

## Begin by installing python 3.10 in the folder:
# for windowsOS:
https://www.python.org/downloads/release/python-3100/
# for linux:
apt install python3.10

## Open a python terminal

## Create a python3.10 virtual environment in the terminal
python3.10 -m venv env 
# Note, you can change "env" to any name

## Navigate into your virtual environment
# Windows:
cd .\env\
# linux:
cd env

## Activate the virtual environment inside the folder
# Windows:
.\bin\activate
# Linux:
source bin/activate

## In the virtual environment:
# Check that the version of python is correct:
python3 --version
# It should say Python 3.10 (3.10.11)
# Now check your virtual environment packages:
pip list
# It should only display pip and setuptools

# Navigate to the include folder of the virtual environment
# Windows:
cd .\include\
# Linux:
cd include/

## Clone the repository
git clone https://github.com/nolans12/Satellite-DDF.git

# Navigate into the repository
# Windows:
cd .\Satellite-DDF\
# Linux:
cd Satellite-DDF/


## Install dependencies
pip install -r requirements.txt

## Navigate to Phase1 Directory
# Windows:
cd .\phase1\
# Linux:
cd phase1/

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
