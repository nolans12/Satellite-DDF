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
# Create a python virtual environment in a python terminal
python3 -m venv env

# Activate the virtual environment
.\env\Scripts\activate

# Clone the repository
git clone https://github.com/nolans12/Satellite-DDF.git

# Navigate to the project directory
cd Satellite-DDF/phase1

# Install python 3.10.0
# for macOS:
brew install python@3.10

# for windowsOS:
https://www.python.org/downloads/release/python-3100/

# Verify the terminal is running the correct version of python

# Install dependencies
pip install -r requirements.txt
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
```
