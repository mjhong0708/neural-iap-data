import ase.build
from ase.calculators.emt import EMT


def get_test_molecule():
    molecule = ase.build.molecule("C6H6")
    calc = EMT()
    molecule.calc = calc
    return molecule


def get_test_bulk(repeat=(1, 1, 1)):
    bulk = ase.build.bulk("Pt", "fcc", a=3.9, cubic=True).repeat(repeat)
    calc = EMT()
    bulk.calc = calc
    return bulk
