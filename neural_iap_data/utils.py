from typing import Optional, Union

import torch
from torch.utils.data import Dataset
from torch_scatter import scatter_add

from neural_iap_data.data.atomsgraph import AtomsGraph

Tensor = torch.Tensor


def remove_extension(filename: str) -> str:
    """Remove file extension."""
    return ".".join(filename.split(".")[:-1])


def n_atoms_per_graph(data: AtomsGraph) -> Tensor:
    """Return the number of atoms per graph."""
    num_nodes = data.pos.size(0)
    return scatter_add(torch.ones(num_nodes), data.batch, dim=0)


def train_val_test_split(
    train_size: Union[int, float],
    val_size: Union[int, float],
    dataset: Dataset = None,
    num_data: int = None,
    return_idx_only: bool = True,
    random_state: Optional[int] = None,
):
    # Input validation
    if dataset is None and num_data is None:
        raise ValueError("Must specify either dataset or num_data.")

    if dataset is not None:
        num_data = len(dataset)

    if not return_idx_only and dataset is None:
        raise ValueError("Must specify dataset if return_idx_only is False.")

    if isinstance(train_size, float):
        train_size = int(train_size * num_data)

    if isinstance(val_size, float):
        val_size = int(val_size * num_data)

    if train_size + val_size > num_data:
        raise ValueError("train_size and val_size are too large.")

    # Split
    lookup = torch.arange(num_data)
    if random_state is not None:
        torch.manual_seed(random_state)
    torch.randperm(num_data, out=lookup)

    train_idx = lookup[:train_size]
    val_idx = lookup[train_size : train_size + val_size]
    test_idx = lookup[train_size + val_size :]

    if return_idx_only:
        return train_idx, val_idx, test_idx
    return dataset[train_idx], dataset[val_idx], dataset[test_idx]
