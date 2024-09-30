## All the libraries that are used in the project are imported here
import copy
import csv
import io
import os
from collections import defaultdict
from copy import deepcopy

import imageio
import jax
import jax.numpy as jnp
import matplotlib.cm as cm  # Import colormap module
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import PIL
import poliastro

# Optimizer:
import pulp
import sympy as sp
from astropy import units as u
from jax import jacfwd, jacrev
from jax.scipy.linalg import expm
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from sympy import Derivative, Function, Matrix, Subs, lambdify, symbols
