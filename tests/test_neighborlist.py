import pytest
import torch
from commons import get_test_bulk, get_test_molecule

from neural_iap_data.neighborlist import ASENeighborListBuilder, MatscipyNeighborListBuilder


@pytest.mark.parametrize(
    "system,self_interaction", [("bulk", False), ("bulk", True), ("molecule", False), ("molecule", True)]
)
def test_matscipy_neighborlist(system, self_interaction):
    if system == "bulk":
        atoms = get_test_bulk()
    elif system == "molecule":
        atoms = get_test_molecule()
    ase_builder = ASENeighborListBuilder(cutoff=6.0, self_interaction=self_interaction)
    matscipy_builder = MatscipyNeighborListBuilder(cutoff=6.0, self_interaction=self_interaction)

    ase_neighborlist = ase_builder.build(atoms)
    matscipy_neighborlist = matscipy_builder.build(atoms)

    for i in range(len(atoms)):
        ase_neighbors_i = ase_neighborlist.neighbor_idx[ase_neighborlist.center_idx == i].sort().values
        matscipy_neighbors_i = matscipy_neighborlist.neighbor_idx[matscipy_neighborlist.center_idx == i].sort().values
        assert torch.allclose(ase_neighbors_i, matscipy_neighbors_i)
