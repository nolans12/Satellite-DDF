## All the libraries that are used in the project are imported here
import poliastro
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import io
import imageio
import os
import PIL
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection