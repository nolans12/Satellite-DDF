# Distributed Data Fusion for Space Based Target Tracking 

## Overview
Simulates cooperative localization in discrete time using satellite agents sensing targets near earth. Each
agent individually estimates the position and velocity of targets in FOV and communicates infromation with 
their neighbors for collective estimation. 

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Contributing](#contributing)
5. [License](#license)
6. [Contact](#contact)

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