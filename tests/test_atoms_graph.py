import torch
from commons import get_test_molecule

from neural_iap_data import AtomsGraph


def test_atoms_graph_free():
    atoms = get_test_molecule()
    atoms_graph = AtomsGraph.from_ase(atoms)

    # test equality
    assert torch.allclose(atoms_graph.pos, torch.as_tensor(atoms.positions, dtype=torch.float))
    assert torch.allclose(atoms_graph.elems, torch.as_tensor(atoms.numbers, dtype=torch.long))
    assert torch.allclose(atoms_graph.cell.squeeze(), torch.as_tensor(atoms.cell.array, dtype=torch.float))
    assert torch.allclose(atoms_graph.batch, torch.zeros_like(atoms_graph.elems, dtype=torch.long))


def test_atoms_graph_neighbor():
    atoms = get_test_molecule()
    atoms_graph = AtomsGraph.from_ase(atoms, build_neighbors=True, cutoff=10.0, self_interaction=False)
    neighbor_idx, center_idx = atoms_graph.edge_index
    for i in range(len(atoms)):
        assert len(neighbor_idx[center_idx == i]) == 11
