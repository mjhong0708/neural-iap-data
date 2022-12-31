import ase.build
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator


def get_test_molecule():
    molecule = ase.build.molecule("C6H6")
    energy = 0.0
    forces = np.zeros_like(molecule.positions)
    calc = SinglePointCalculator(molecule, energy=energy, forces=forces)
    molecule.calc = calc
    return molecule


def get_test_bulk(repeat=(1, 1, 1)):
    bulk = ase.build.bulk("Pt", "fcc", a=3.9, cubic=True).repeat(repeat)
    energy = 0.0
    forces = np.zeros_like(bulk.positions)
    calc = SinglePointCalculator(bulk, energy=energy, forces=forces)
    bulk.calc = calc
    return bulk
