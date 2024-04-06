"""SupraMolecule class for optimisation."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import mchammer as mch
import networkx as nx
import numpy as np

if typing.TYPE_CHECKING:
    from collections import abc


@dataclass
class SupraMolecule(mch.Molecule):
    """Representation of a supramolecule containing atoms and positions.

    Parameters:
        atoms : :class:`iterable` of :class:`.Atom`
            Atoms that define the molecule.

        bonds : :class:`iterable` of :class:`.Bond`
            Bonds between atoms that define the molecule.

        position_matrix : :class:`numpy.ndarray`
            A ``(n, 3)`` matrix holding the position of every atom in
            the :class:`.Molecule`.

        cid : :class:`int`, optional
            Conformer id of supramolecule.

        potential : :class:`float`, optional
            Potential energy of Supramolecule.

    """

    atoms: tuple[mch.Atom, ...]
    bonds: tuple[mch.Bond, ...]
    position_matrix: np.ndarray
    cid: int | None = None
    potential: float | None = None

    def __post_init__(self) -> None:
        """Post initialization of molecule."""
        self.atoms = tuple(self.atoms)
        self.bonds = tuple(self.bonds)
        self.position_matrix = np.array(
            self.position_matrix.T,
            dtype=np.float64,
        )

        self._define_components()

    def with_position_matrix(
        self,
        position_matrix: np.ndarray,
    ) -> SupraMolecule:
        """Return clone SupraMolecule with new position matrix.

        Parameters:
            position_matrix:
                A position matrix of the clone. The shape of the matrix
                is ``(n, 3)``.

        """
        _temp_components = tuple(self.get_components())

        _temp_supramolecule = SupraMolecule(
            atoms=tuple(self.atoms),
            bonds=tuple(self.bonds),
            position_matrix=np.array(position_matrix),
            cid=self.cid,
            potential=self.potential,
        )
        # Overwrite redefined components.
        _temp_supramolecule.components = _temp_components
        return _temp_supramolecule

    def with_displacement(self, displacement: np.ndarray) -> SupraMolecule:
        """Return a displaced clone Molecule.

        Parameters:
            displacement:
                The displacement vector to be applied.

        """
        new_position_matrix = self.position_matrix.T + displacement
        return SupraMolecule(
            atoms=tuple(self.atoms),
            bonds=tuple(self.bonds),
            cid=self.cid,
            potential=self.potential,
            position_matrix=np.array(new_position_matrix),
        )

    @classmethod
    def init_from_components(
        cls,  # noqa: ANN102
        components: list[mch.Molecule],
        cid: int | None = None,
        potential: float | None = None,
    ) -> typing.Self:
        """Initialize a :class:`Supramolecule` instance from components.

        Parameters:
            components:
                Molecular components that define the supramolecule.

            cid:
                Conformer id of supramolecule.

            potential:
                Potential energy of Supramolecule.

        """
        atoms = []
        bonds = []
        position_matrix = []
        # Map old atom ids in components to atom ids in supramolecule.
        atom_id_map: dict[int, int] = {}
        bond_id_map: dict[int, int] = {}
        for comp in components:
            for a in comp.get_atoms():
                if len(atom_id_map) == 0:
                    atom_id_map[a.get_id()] = 0
                else:
                    atom_id_map[a.get_id()] = (
                        max(list(atom_id_map.values())) + 1
                    )
                atoms.append(
                    mch.Atom(
                        id=atom_id_map[a.get_id()],
                        element_string=a.get_element_string(),
                    )
                )
            for b in comp.get_bonds():
                if len(bond_id_map) == 0:
                    bond_id_map[b.get_id()] = 0
                else:
                    bond_id_map[b.get_id()] = (
                        max(list(bond_id_map.values())) + 1
                    )
                bonds.append(
                    mch.Bond(
                        id=bond_id_map[b.get_id()],
                        atom_ids=(
                            atom_id_map[b.get_atom1_id()],
                            atom_id_map[b.get_atom2_id()],
                        ),
                    )
                )
            for pos in comp.get_position_matrix():
                position_matrix.append(pos)  # noqa: PERF402

        supramolecule: SupraMolecule = cls.__new__(cls)
        supramolecule.atoms = tuple(atoms)
        supramolecule.bonds = tuple(bonds)
        supramolecule.components = tuple(components)
        supramolecule.cid = cid
        supramolecule.potential = potential
        supramolecule.position_matrix = np.array(position_matrix).T
        return supramolecule

    def _define_components(self) -> None:
        """Define disconnected component molecules as :class:`.Molecule`s."""
        # Produce a graph from the molecule that does not include edges
        # where the bonds to be optimized are.
        mol_graph = nx.Graph()
        for atom in self.get_atoms():
            mol_graph.add_node(atom.get_id())

        # Add edges.
        for bond in self.bonds:
            pair_ids = (bond.get_atom1_id(), bond.get_atom2_id())
            mol_graph.add_edge(*pair_ids)

        # Get atom ids in disconnected subgraphs.
        comps = []
        for c in nx.connected_components(mol_graph):
            c_ids = sorted(c)
            in_atoms = [i for i in self.atoms if i.get_id() in c]
            in_bonds = [
                i
                for i in self.bonds
                if i.get_atom1_id() in c and i.get_atom2_id() in c
            ]
            new_pos_matrix = self.position_matrix[:, list(c_ids)].T
            comps.append(mch.Molecule(in_atoms, in_bonds, new_pos_matrix))

        self.components = tuple(comps)

    def write_xyz_content(self) -> list[str]:
        """Write basic `.xyz` file content of Molecule."""
        coords = self.get_position_matrix()
        content = ["0"]
        for i, atom in enumerate(self.get_atoms(), 1):
            x, y, z = (i for i in coords[atom.get_id()])
            content.append(f"{atom.get_element_string()} {x:f} {y:f} {z:f}\n")
        # Set first line to the atom_count.
        content[0] = f"{i}\ncid:{self.cid}, pot: {self.potential}\n"

        return content

    def get_components(self) -> abc.Iterable[mch.Molecule]:
        """Yields each molecular component."""
        yield from self.components

    def get_cid(self) -> int | None:
        """Get conformer id."""
        return self.cid

    def get_potential(self) -> float | None:
        """Get potential energy."""
        return self.potential

    def __str__(self) -> str:
        """String representation of SupraMolecule."""
        return repr(self)

    def __repr__(self) -> str:
        """String representation of SupraMolecule."""
        comps = ", ".join([str(i) for i in self.get_components()])
        return (
            f"{self.__class__.__name__}("
            f"{len(list(self.get_components()))} components, "
            f"{comps})"
        )
