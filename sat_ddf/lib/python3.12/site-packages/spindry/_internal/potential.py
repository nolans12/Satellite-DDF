"""Classes for calculating the potential energy of supramolecules."""

from __future__ import annotations

import itertools as it
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from .supramolecule import SupraMolecule


class Potential:
    """Base class for potential calculators."""

    def compute_potential(self, supramolecule: SupraMolecule) -> float:
        """Calculate potential energy.

        Parameters:
            supramolecular:
                Supramolecule to evaluate.
        """
        raise NotImplementedError


class SpdPotential(Potential):
    """Default spindry non-bonded potential function."""

    def __init__(self, nonbond_epsilon: float = 5) -> None:
        """Initialize a :class:`Spinner` instance.

        Parameters:
            nonbond_epsilon:
                Value of epsilon used in the nonbond potential in MC moves.
                Determines strength of the nonbond potential.
                Defaults to 5.

        """
        self._nonbond_epsilon = nonbond_epsilon

    def _nonbond_potential(
        self,
        distance: np.ndarray,
        sigmas: np.ndarray,
    ) -> np.ndarray:
        """Define a Lennard-Jones nonbonded potential.

        This potential has no relation to an empircal forcefield.

        """
        return self._nonbond_epsilon * (
            (sigmas / distance) ** 12 - (sigmas / distance) ** 6
        )

    def _combine_sigma(self, radii1: tuple, radii2: tuple) -> np.ndarray:
        """Combine radii using Lorentz-Berthelot rules."""
        len1 = len(radii1)
        len2 = len(radii2)

        mixed = np.zeros((len1, len2))
        for i in range(len1):
            for j in range(len2):
                mixed[i, j] = (radii1[i] + radii2[j]) / 2

        return mixed

    def _compute_nonbonded_potential(
        self,
        position_matrices: list[np.ndarray],
        radii: list[tuple],
    ) -> float:
        nonbonded_potential = 0
        for pos_mat_pair, radii_pair in zip(
            it.combinations(position_matrices, 2),
            it.combinations(radii, 2),
            strict=True,
        ):
            pair_dists = cdist(pos_mat_pair[0], pos_mat_pair[1])
            sigmas = self._combine_sigma(radii_pair[0], radii_pair[1])
            nonbonded_potential += np.sum(
                self._nonbond_potential(
                    distance=pair_dists.flatten(),
                    sigmas=sigmas.flatten(),
                )
            )

        return nonbonded_potential

    def compute_potential(self, supramolecule: SupraMolecule) -> float:
        """Compute the potential energy of a supramolecule."""
        component_position_matrices = [
            i.get_position_matrix() for i in supramolecule.get_components()
        ]
        component_radii = [
            tuple(j.get_radius() for j in i.get_atoms())
            for i in supramolecule.get_components()
        ]
        return self._compute_nonbonded_potential(
            position_matrices=component_position_matrices,
            radii=component_radii,
        )


class VaryingEpsilonPotential(Potential):
    """A non-bonded potential function with varying epsilons."""

    def _nonbond_potential(
        self,
        distance: np.ndarray,
        sigmas: np.ndarray,
        epsilons: np.ndarray,
    ) -> np.ndarray:
        """Define a Lennard-Jones nonbonded potential.

        This potential has no relation to an empircal forcefield.

        """
        return epsilons * (
            (sigmas / distance) ** 12 - (sigmas / distance) ** 6
        )

    def _combine_sigma(self, radii1: tuple, radii2: tuple) -> np.ndarray:
        """Combine radii using Lorentz-Berthelot rules."""
        len1 = len(radii1)
        len2 = len(radii2)

        mixed = np.zeros((len1, len2))
        for i in range(len1):
            for j in range(len2):
                mixed[i, j] = (radii1[i] + radii2[j]) / 2

        return mixed

    def _combine_epsilon(self, e1: tuple, e2: tuple) -> np.ndarray:
        """Combine epsilon using Lorentz-Berthelot rules."""
        len1 = len(e1)
        len2 = len(e2)

        mixed = np.zeros((len1, len2))
        for i in range(len1):
            for j in range(len2):
                mixed[i, j] = np.sqrt(e1[i] * e2[j])

        return mixed

    def _compute_nonbonded_potential(
        self,
        position_matrices: list[np.ndarray],
        radii: list[tuple],
        epsilons: list[tuple],
    ) -> float:
        nonbonded_potential = 0
        for pos_mat_pair, radii_pair, epsilon_pair in zip(
            it.combinations(position_matrices, 2),
            it.combinations(radii, 2),
            it.combinations(epsilons, 2),
            strict=True,
        ):
            pair_dists = cdist(pos_mat_pair[0], pos_mat_pair[1])
            new_radii = self._combine_sigma(radii_pair[0], radii_pair[1])
            new_epsilons = self._combine_epsilon(
                epsilon_pair[0], epsilon_pair[1]
            )
            nonbonded_potential += np.sum(
                self._nonbond_potential(
                    distance=pair_dists.flatten(),
                    sigmas=new_radii.flatten(),
                    epsilons=new_epsilons.flatten(),
                )
            )

        return nonbonded_potential

    def compute_potential(self, supramolecule: SupraMolecule) -> float:
        """Compure the potential of the molecule."""
        component_position_matrices = [
            i.get_position_matrix() for i in supramolecule.get_components()
        ]
        component_radii = [
            tuple(j.sigma for j in i.get_atoms())
            for i in supramolecule.get_components()
        ]
        component_epsilon = [
            tuple(j.epsilon for j in i.get_atoms())
            for i in supramolecule.get_components()
        ]
        return self._compute_nonbonded_potential(
            position_matrices=component_position_matrices,
            radii=component_radii,
            epsilons=component_epsilon,
        )
