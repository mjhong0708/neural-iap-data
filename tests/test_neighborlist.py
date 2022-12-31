import pytest
import torch
from commons import get_test_bulk, get_test_molecule
from torch_geometric.utils import sort_edge_index

from neural_iap_data.data import AtomsGraph


@pytest.mark.parametrize(
    "system,self_interaction", [("bulk", False), ("bulk", True), ("molecule", False), ("molecule", True)]
)
def test_matscipy_neighborlist(system, self_interaction):
    cutoff = 5.0
    if system == "bulk":
        atoms = get_test_bulk(repeat=(3, 3, 3))
    elif system == "molecule":
        atoms = get_test_molecule()

    graph_ase = AtomsGraph.from_ase(atoms, True, cutoff, self_interaction, neighborlist_backend="ase")
    graph_mat = AtomsGraph.from_ase(atoms, True, cutoff, self_interaction, neighborlist_backend="matscipy")
    assert is_neighborlist_identical(graph_ase, graph_mat)


def is_neighborlist_identical(graph_1, graph_2):
    edge_index_1, edge_shift_1 = sort_edge_index(graph_1.edge_index, graph_1.edge_shift, sort_by_row=False)
    edge_index_2, edge_shift_2 = sort_edge_index(graph_2.edge_index, graph_2.edge_shift, sort_by_row=False)
    graph_1 = graph_1.clone()
    graph_2 = graph_2.clone()
    graph_1.edge_index, graph_1.edge_shift = edge_index_1, edge_shift_1
    graph_2.edge_index, graph_2.edge_shift = edge_index_2, edge_shift_2

    # Check the index of neighbors
    if not torch.allclose(graph_1.edge_index, graph_2.edge_index):
        return False

    # Check the distance to neighbors
    vec_1 = graph_1.compute_edge_vecs()
    vec_2 = graph_2.compute_edge_vecs()
    d_1 = vec_1.norm(dim=1)
    d_2 = vec_2.norm(dim=1)
    return torch.allclose(d_1, d_2)
