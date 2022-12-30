import os.path as osp
from typing import Callable, Literal, Optional

import numpy as np
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from torch_geometric.data import download_url, extract_tar, extract_zip
from tqdm import tqdm

from neural_iap_data.data import AtomsDataset

kcal_mol_to_ev = units.kcal / units.mol


class MD17Dataset(AtomsDataset):
    urls = {
        "benzene": "http://www.quantum-machine.org/gdml/data/npz/benzene2017_dft.npz",
        "uracil": "http://www.quantum-machine.org/gdml/data/npz/uracil_dft.npz",
        "naphthalene": "http://www.quantum-machine.org/gdml/data/npz/naphthalene_dft.npz",
        "aspirin": "http://www.quantum-machine.org/gdml/data/npz/aspirin_dft.npz",
        "salicylic_acid": "http://www.quantum-machine.org/gdml/data/npz/salicylic_dft.npz",
        "malonaldehyde": "http://www.quantum-machine.org/gdml/data/npz/malonaldehyde_dft.npz",
        "ethanol": "http://www.quantum-machine.org/gdml/data/npz/ethanol_dft.npz",
        "toluene": "http://www.quantum-machine.org/gdml/data/npz/toluene_dft.npz",
        "paracetamol": "http://www.quantum-machine.org/gdml/data/npz/paracetamol_dft.npz",
        "azobenzene": "http://www.quantum-machine.org/gdml/data/npz/azobenzene_dft.npz",
        "aspirin_ccsd": "http://www.quantum-machine.org/gdml/data/npz/aspirin_ccsd.zip",
        "benzene_ccsd_t": "http://www.quantum-machine.org/gdml/data/npz/benzene_ccsd_t.zip",
        "malonaldehyde_ccsd_t": "http://www.quantum-machine.org/gdml/data/npz/malonaldehyde_ccsd_t.zip",
        "toluene_ccsd_t": "http://www.quantum-machine.org/gdml/data/npz/toluene_ccsd_t.zip",
        "ethanol_ccsd_t": "http://www.quantum-machine.org/gdml/data/npz/ethanol_ccsd_t.zip",
    }
    energy_key = "E"
    force_key = "F"
    positions_key = "R"
    elems_key = "z"

    def __init__(
        self,
        name: str,
        suffix: str = "",
        cutoff: float = 5.0,
        unit: Literal["kcal/mol", "eV"] = "kcal/mol",
        num_workers: int = 4,
        shift_energy: bool = True,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
    ):
        self.suffix = suffix
        self.root = root
        self.unit = unit
        self.num_workers = num_workers
        if "ccsd_t" in name:
            url = self.urls[name.replace("_train", "").replace("_test", "")]
        else:
            url = self.urls[name]
        path = download_url(url, self.root)
        if "ccsd_t" in name:
            extract_zip(path, self.root)
            if name.endswith("train") or name.endswith("test"):
                filename = osp.join(self.root, name.replace("_train", "-train").replace("_test", "-test") + ".npz")
            else:
                raise ValueError("You must specify train or test for ccsd_t datasets.")
            atoms_list = self._parse_file(filename)
        else:
            atoms_list = self._parse_file(path)
        self.name = name + "_" + suffix
        super().__init__(
            self.name,
            cutoff,
            atoms_list,
            num_workers,
            shift_energy,
            root,
            transform,
            pre_transform,
            pre_filter,
            log,
        )
        if unit != self.unit:
            print("Unit is not consistent with the original dataset. Please check the unit.")
        self.unit = unit

    @classmethod
    def available_datasets(cls):
        return list(cls.urls.keys())

    def _download(self) -> str:
        url = MD17Dataset.urls[self.name]
        path = download_url(url, self.root)
        if self.name.endswith("ccsd_t"):
            path = extract_zip(path, self.root)

    def _parse_file(self, filename):
        """Parse a MD17 file into a list of Atoms objects."""
        data = np.load(filename)
        conversion = 1.0 if self.unit == "kcal/mol" else kcal_mol_to_ev
        E = data[self.energy_key].squeeze() * conversion
        F = data[self.force_key].squeeze() * conversion
        R = data[self.positions_key].squeeze()
        z = data[self.elems_key].squeeze()

        atoms_list = []
        for i in tqdm(range(len(E)), desc="Parsing MD17 file"):
            atoms = Atoms(
                numbers=z,
                positions=R[i],
                pbc=False,
            )
            atoms.set_calculator(SinglePointCalculator(atoms, energy=E[i], forces=F[i]))
            atoms_list.append(atoms)
        return atoms_list


class MD22Dataset(MD17Dataset):
    urls = {
        "Ac-Ala3-NHMe": "http://www.quantum-machine.org/gdml/repo/datasets/md22_Ac-Ala3-NHMe.npz",
        "DHA": "http://www.quantum-machine.org/gdml/repo/datasets/md22_DHA.npz",
        "Stachyose": "http://www.quantum-machine.org/gdml/repo/datasets/md22_stachyose.npz",
        "AT-AT": "http://www.quantum-machine.org/gdml/repo/datasets/md22_AT-AT.npz",
        "AT-AT-CG-CG": "http://www.quantum-machine.org/gdml/repo/datasets/md22_AT-AT-CG-CG.npz",
        "Buckyball_catcher": "http://www.quantum-machine.org/gdml/repo/datasets/md22_buckyball-catcher.npz",
        "DWNT": "http://www.quantum-machine.org/gdml/repo/datasets/md22_double-walled_nanotube.npz",
    }


class RevisedMD17Dataset(MD17Dataset):
    url = r"https://archive.materialscloud.org/record/file?filename=rmd17.tar.bz2&record_id=466"
    filename = "rmd17.tar.bz2"
    npz_dir = "rmd17/npz_data"
    split_dir = "rmd17/splits"
    energy_key = "energies"
    force_key = "forces"
    positions_key = "coords"
    elems_key = "nuclear_charges"
    file_map = {
        "benzene": "rmd17_benzene.npz",
        "uracil": "rmd17_uracil.npz",
        "naphthalene": "rmd17_naphthalene.npz",
        "aspirin": "rmd17_aspirin.npz",
        "salicylic_acid": "rmd17_salicylic.npz",
        "malonaldehyde": "rmd17_malonaldehyde.npz",
        "ethanol": "rmd17_ethanol.npz",
        "toluene": "rmd17_toluene.npz",
        "paracetamol": "rmd17_paracetamol.npz",
        "azobenzene": "rmd17_azobenzene.npz",
    }

    def __init__(
        self,
        name: str,
        suffix: str = "",
        split: str = "train_1",
        cutoff: float = 5.0,
        unit: Literal["kcal/mol", "eV"] = "kcal/mol",
        num_workers: int = 4,
        shift_energy: bool = True,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
    ):
        self.name = name + "_" + split + "_" + suffix
        self.suffix = suffix
        self.split = split
        self.root = root
        self.unit = unit
        self.num_workers = num_workers
        path = download_url(self.url, self.root, filename="rmd17.tar.bz2")
        if not osp.isdir(osp.join(self.root, self.npz_dir)):
            extract_tar(path, self.root, mode="r:bz2")
        try:
            selected_file = self.file_map[name]
        except KeyError as e:
            raise ValueError(f"Invalid dataset name: {name}") from e
        self.split_idx = {}
        for split in ("train", "test"):
            for i in range(1, 5 + 1):
                splitfile = osp.join(self.root, self.split_dir, f"index_{split}_0{i}.csv")
                self.split_idx[f"{split}_{i}"] = np.loadtxt(splitfile).astype(np.int64).squeeze()
        if osp.isfile(osp.join(self.root, "processed", self.name + ".pt")):
            atoms_list = None
        else:
            atoms_list = self._parse_file(osp.join(self.root, self.npz_dir, selected_file))
            atoms_list = [atoms_list[i] for i in self.split_idx[self.split]]
        AtomsDataset.__init__(
            self,
            self.name,
            cutoff,
            atoms_list,
            num_workers,
            shift_energy,
            root,
            transform,
            pre_transform,
            pre_filter,
            log,
        )
        if unit != self.unit:
            print("Unit is not consistent with the original dataset. Please check the unit.")
        self.unit = unit
