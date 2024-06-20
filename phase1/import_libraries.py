## All the libraries that are used in the project are imported here
import poliastro
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import io
import imageio
import os
import PIL
import csv
import copy
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax import jacfwd, jacrev
import sympy as sp
from sympy import Matrix, symbols, lambdify, Subs, Derivative, Function
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.cm as cm  # Import colormap module
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag
from collections import defaultdict
from ambiance import Atmosphere # NOLAN install this libray for Std Atm