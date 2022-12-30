from abc import ABC, abstractmethod
from typing import NamedTuple, Union

import ase.neighborlist
import matscipy.neighbours
import torch
from ase import Atoms
from torch import Tensor
import numpy as np


class NeighborList(NamedTuple):
    center_idx: Tensor
    neighbor_idx: Tensor
    offset: Tensor


class NeighborListBuilder(ABC):
    def __init__(self, cutoff: float, self_interaction: bool = False):
        self.cutoff = cutoff
        self.self_interaction = self_interaction

    @abstractmethod
    def build(self, atoms: Atoms) -> NeighborList:
        """Build neighbor list for given atoms."""


class ASENeighborListBuilder(NeighborListBuilder):
    def build(self, atoms: Atoms) -> NeighborList:
        center_idx, neighbor_idx, offset = ase.neighborlist.neighbor_list(
            "ijS",
            a=atoms,
            cutoff=self.cutoff,
            self_interaction=self.self_interaction,
        )
        return NeighborList(
            center_idx=torch.LongTensor(center_idx),
            neighbor_idx=torch.LongTensor(neighbor_idx),
            offset=torch.as_tensor(offset, dtype=torch.float),
        )


class MatscipyNeighborListBuilder(NeighborListBuilder):
    def build(self, atoms: Atoms) -> NeighborList:
        # matscipy.neighbours.neighbour_list fails for non-periodic systems
        if not atoms.pbc.all():
            # put atoms in a box with periodic boundary conditions
            atoms = atoms.copy()
            rmin = np.min(atoms.positions, axis=0)
            rmax = np.max(atoms.positions, axis=0)
            celldim = np.max(rmax - rmin) + 2.5 * self.cutoff
            atoms.cell = np.eye(3) * celldim
            atoms.pbc = True
            atoms.center()

        center_idx, neighbor_idx, offset = matscipy.neighbours.neighbour_list("ijS", atoms, self.cutoff)
        # add self interaction as ase.neighborlist does
        if self.self_interaction:
            center_idx = np.concatenate([center_idx, np.arange(len(atoms))])
            neighbor_idx = np.concatenate([neighbor_idx, np.arange(len(atoms))])
            offset = np.concatenate([offset, np.zeros((len(atoms), 3))])
            # sort by center_idx
            idx = np.argsort(center_idx)
            center_idx = center_idx[idx]
            neighbor_idx = neighbor_idx[idx]
            offset = offset[idx]

        return NeighborList(
            center_idx=torch.LongTensor(center_idx),
            neighbor_idx=torch.LongTensor(neighbor_idx),
            offset=torch.as_tensor(offset, dtype=torch.float),
        )


_neighborlistbuilder_cls_map = {"ase": ASENeighborListBuilder, "matscipy": MatscipyNeighborListBuilder}


def resolve_neighborlist_builder(neighborlist_backend: Union[str, object]) -> object:
    if isinstance(neighborlist_backend, str):
        return _neighborlistbuilder_cls_map[neighborlist_backend]
    return neighborlist_backend
