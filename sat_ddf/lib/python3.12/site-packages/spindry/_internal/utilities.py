"""This module defines general-purpose objects, functions and classes."""

import itertools as it

import numpy as np
from scipy.spatial.distance import cdist, euclidean

from .supramolecule import SupraMolecule


def get_atom_distance(
    position_matrix: np.ndarray,
    atom1_id: int,
    atom2_id: int,
) -> float:
    """Return the distance between two atoms."""
    return float(
        euclidean(u=position_matrix[atom1_id], v=position_matrix[atom2_id])
    )


def calculate_min_atom_distance(supramolecule: SupraMolecule) -> float:
    """Calculate the minimum distance between components in supramolecule."""
    component_position_matrices = (
        i.get_position_matrix() for i in supramolecule.get_components()
    )

    min_distance = 1e24
    for pos_mat_pair in it.combinations(component_position_matrices, 2):
        pair_dists = cdist(pos_mat_pair[0], pos_mat_pair[1])
        min_distance = min([min_distance, min(pair_dists.flatten())])

    return min_distance


def calculate_centroid_distance(supramolecule: SupraMolecule) -> float:
    """Calculate the centroid distances in 1:1 complex."""
    comps = list(supramolecule.get_components())
    if len(comps) != 2:  # noqa: PLR2004
        msg = "more than one guest there buddy!"
        raise ValueError(msg)

    return float(
        np.linalg.norm(comps[0].get_centroid() - comps[1].get_centroid())
    )
