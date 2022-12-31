"""
Data structure for constructing dataset for atomistic machine learning.
"""
from typing import Sequence, Union

import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor, Tensor

from neural_iap_data import utils
from neural_iap_data.neighborlist import NeighborListBuilder, resolve_neighborlist_builder

IndexType = Union[slice, Tensor, np.ndarray, Sequence]
_default_dtype = torch.get_default_dtype()


class AtomsGraph(Data):
    """Basic graph representation of an atomic system.

    Args:
        elems (Tensor): 1D tensor of atomic numbers.
        pos (Tensor): 2D tensor of atomic positions. (N, 3)
        cell (OptTensor, optional): 1D or 2D tensor of lattice vectors. (3, 3) or (1, 3, 3)
            Automatically unsqueeze to 2D if 1D is given.
        edge_index (OptTensor, optional): Edge index. Defaults to None.
            If this means neighbor indices, 0th row is neighbor and 1st row is center.
            This is because message passing occurs from neighbors to centers.
        edge_shift (OptTensor, optional): Optional shift vectors when creating neighbor list.
            Defaults to None.
        energy (OptTensor, optional): Energy of the system. Defaults to None.
        force (OptTensor, optional): Force on each atom. Defaults to None.
        node_features (OptTensor, optional): Node features. Defaults to None.
        edge_features (OptTensor, optional): Edge features. Defaults to None.
        global_features (OptTensor, optional): Global features. Defaults to None.
        add_batch (bool, optional): If True, add batch index to the graph. Defaults to False.
    """

    def __init__(
        self,
        elems: Tensor = None,
        pos: Tensor = None,
        cell: OptTensor = None,
        edge_index: OptTensor = None,
        edge_shift: OptTensor = None,
        energy: OptTensor = None,
        force: OptTensor = None,
        node_features: OptTensor = None,
        edge_features: OptTensor = None,
        global_features: OptTensor = None,
        add_batch: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.elems = elems
        self.pos = pos
        self.cell = cell
        self.edge_index = edge_index
        self.edge_shift = edge_shift
        self.energy = energy
        self.force = force
        self.node_features = node_features
        self.edge_features = edge_features
        self.global_features = global_features
        if add_batch:
            self.batch = torch.zeros_like(elems, dtype=torch.long, device=pos.device)

    @classmethod
    def from_ase(
        cls,
        atoms: Atoms,
        build_neighbors: bool = False,
        cutoff: float = 5.0,
        self_interaction: bool = False,
        energy: float = None,
        force: Tensor = None,
        *,
        add_batch: bool = True,
        neighborlist_backend: Union[str, NeighborListBuilder] = "ase",
        **kwargs,
    ):
        """Create AtomsGraph from ASE Atoms object.

        Args:
            atoms (Atoms): An ASE Atoms object.
            build_neighbors (bool, optional): Whether to build neighborlist or not. Defaults to False.
            cutoff (float, optional): Cutoff radius for neighbor list in Angstrom. Defaults to 5.0.
            self_interaction (bool, optional): Whether to add atom as neighbor of itself(=self loop). Defaults to False.
            energy (float, optional): Potential energy of the system. When set to None, energy will retrieved from
                    atoms.calc if available. Defaults to None.
            force (Tensor, optional): Interatomic forces of the system. When set to None, forces will retrieved from
                    atoms.calc if available. Defaults to None.
            add_batch (bool, optional): Whether to add batch index as attribute or not. Defaults to True.
            neighborlist_backend (Union[str, NeighborListBuilder], optional): The backend for building neighborlist.
                    Accepts `str` or `NeighborListBuilder` class. See `neural_iap_data.neighborlist.py`.
                    Defaults to "ase".

        Returns:
            _type_: _description_
        """

        elems = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=_default_dtype)
        cell = cls.resolve_cell(atoms)

        if energy is None:
            try:
                energy = atoms.get_potential_energy()
                energy = torch.as_tensor(energy, dtype=_default_dtype)
            except RuntimeError:
                pass
        if force is None:
            try:
                force = atoms.get_forces()
                force = torch.as_tensor(force, dtype=_default_dtype)
            except RuntimeError:
                pass

        atoms_graph = cls(elems, pos, cell, None, None, energy, force, add_batch=add_batch, **kwargs)
        if build_neighbors:
            neighborlist_builder_cls = resolve_neighborlist_builder(neighborlist_backend)
            neighborlist_builder: NeighborListBuilder = neighborlist_builder_cls(cutoff, self_interaction)
            center_idx, neigh_idx, edge_shift = neighborlist_builder.build(atoms_graph)
            # edge index: [dst, src]
            edge_index = torch.stack([neigh_idx, center_idx], dim=0)
            atoms_graph.edge_index = edge_index
            atoms_graph.edge_shift = edge_shift
        return atoms_graph

    def to_ase(self) -> Atoms:
        """Convert to Atoms object."""
        if self.cell.norm().abs() < 1e-6:
            pbc = False
        else:
            pbc = True
        atoms = Atoms(
            numbers=self.elems.cpu().numpy(),
            positions=self.pos.cpu().numpy(),
            cell=self.cell.cpu().numpy()[0] if pbc else None,
            pbc=pbc,
        )
        return atoms

    def volume(self) -> Tensor:
        """Return volume of the cell."""
        return self.cell.squeeze().det()

    def n_atoms(self) -> int:
        """Return number of atoms."""
        return utils.n_atoms_per_graph(self).squeeze()

    def compute_edge_vecs(self):
        if "edge_index" not in self:
            raise ValueError("Neighbor list is not built.")
        pos = self.pos
        batch = self.batch
        dst, src = self.edge_index
        edge_batch = batch[src]  # batch index for edges(neighbors)
        vec = pos[dst] - pos[src]
        cell = self.cell if "cell" in self else torch.zeros((batch.max() + 1, 3, 3)).to(vec.device)
        vec += torch.einsum("ni,nij->nj", self.edge_shift, cell[edge_batch])
        return vec

    @staticmethod
    def resolve_cell(atoms: Atoms) -> Tensor:
        """Resolve cell as tensor from Atoms object."""
        # reject partial pbc
        if atoms.pbc.any() and not atoms.pbc.all():
            raise ValueError("AtomsGraph does not support partial pbc")
        pbc = atoms.pbc.all()
        if pbc:
            return torch.tensor(atoms.cell.array, dtype=_default_dtype).unsqueeze(0)
        # Return zeros when pbc is false
        return torch.zeros((1, 3, 3), dtype=_default_dtype)
