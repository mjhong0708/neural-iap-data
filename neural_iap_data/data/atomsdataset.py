"""
Data structure for constructing dataset for atomistic machine learning.
"""
import os.path as osp
import sys
import warnings
from typing import Callable, Optional, Sequence, Union

import ase.io
import numpy as np
import torch
from ase import Atoms
from joblib import Parallel, delayed
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.dataset import _repr, files_exist
from torch_geometric.data.makedirs import makedirs
from torch_geometric.typing import Tensor
from tqdm import tqdm

from .atomsdata import AtomsGraph
from ..utils import remove_extension

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


class AtomsDataset(InMemoryDataset):
    def __init__(
        self,
        name: str,
        cutoff: float = 5.0,
        atoms_list: list[Atoms] = None,
        num_workers: int = 4,
        shift_energy: bool = True,  # if true, shifted as energy - energy.max()
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        *,
        neighborlist_backend: str = "ase",
    ):
        self.name = name
        self.cutoff = cutoff
        self.atoms_list = atoms_list
        self.num_workers = num_workers
        self.neighborlist_backend = neighborlist_backend
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.shift_energy = shift_energy
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.shift_energy:
            energy_max = self.data.energy.max()
            self.data.energy -= energy_max

    @property
    def processed_file_names(self):
        return [f"{self.name}.pt"]

    def process(self):
        # Read data into huge `Data` list.
        def to_graph(atoms, cutoff):
            return AtomsGraph.from_ase(atoms, True, cutoff, neighborlist_backend=self.neighborlist_backend)

        if self.num_workers == 1:
            data_list = [to_graph(atoms, self.cutoff) for atoms in tqdm(self.atoms_list)]
        else:
            data_list = Parallel(n_jobs=self.num_workers, verbose=10)(
                delayed(to_graph)(atoms, self.cutoff) for atoms in self.atoms_list
            )

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union["AtomsDataset", AtomsGraph]:
        item = super().__getitem__(idx)
        if isinstance(item, AtomsGraph):
            item.batch = torch.zeros_like(item.elems, dtype=torch.long, device=item.pos.device)
        return item

    def _process(self):
        pre_transform_filename = f"{self.name}_pre_transform.pt"
        pre_filter_filename = f"{self.name}_pre_filter.pt"
        f = osp.join(self.processed_dir, pre_transform_filename)
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first"
            )

        f = osp.join(self.processed_dir, pre_filter_filename)
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, make sure to "
                "delete '{self.processed_dir}' first"
            )

        if files_exist(self.processed_paths):  # pragma: no cover
            msg = (
                "Using processed data, input arguments other then name does not take effect. "
                "If you want to re-process, delete the files in the processed folder."
            )
            print(msg, file=sys.stderr)
            return

        if self.log:
            print("Processing...", file=sys.stderr)

        makedirs(self.processed_dir)
        self.process()

        path = osp.join(self.processed_dir, pre_transform_filename)
        torch.save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, pre_filter_filename)
        torch.save(_repr(self.pre_filter), path)

        if self.log:
            print("Done!", file=sys.stderr)

    @classmethod
    def from_file(cls, filename: str, name=None, index=":", root="data", cutoff=5.0, **kwargs) -> "AtomsDataset":
        """Create dataset from file."""
        atoms_list = ase.io.read(filename, index=index)
        name = remove_extension(osp.basename(filename)) if name is None else name
        return cls(name=name, atoms_list=atoms_list, root=root, cutoff=cutoff, **kwargs)


