import torch
from commons import get_test_bulk, get_test_molecule
from torch_geometric.data import Batch

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


def test_atoms_graph_n_atoms():
    atoms_1 = get_test_bulk()
    atoms_2 = atoms_1.repeat((2, 2, 2))
    atoms_2.calc = atoms_1.calc
    g1 = AtomsGraph.from_ase(atoms_1)
    g2 = AtomsGraph.from_ase(atoms_2)
    batch = Batch.from_data_list([g1, g2])
    n_atoms = batch.n_atoms
    assert n_atoms[0] == len(atoms_1)
    assert n_atoms[1] == len(atoms_2)


def test_atoms_graph_volume():
    atoms_1 = get_test_bulk()
    atoms_2 = atoms_1.repeat((2, 2, 2))
    atoms_2.calc = atoms_1.calc
    g1 = AtomsGraph.from_ase(atoms_1)
    g2 = AtomsGraph.from_ase(atoms_2)
    batch = Batch.from_data_list([g1, g2])
    volume = batch.volume()
    assert torch.allclose(volume[0], torch.as_tensor(atoms_1.get_volume(), dtype=torch.float))
    assert torch.allclose(volume[1], torch.as_tensor(atoms_2.get_volume(), dtype=torch.float))
