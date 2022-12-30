from torch_geometric.transforms import BaseTransform


class ScaleProperty(BaseTransform):
    r"""Scales the properties of a graph by a given factor.
    Args:
        keys (list of str): The keys of the properties to scale.
        factor (float): The scaling factor.
    """

    def __init__(self, key, factor):
        self.key = key
        self.factor = factor

    def __call__(self, data):
        data[self.key] *= self.factor
        return data

    def __repr__(self):
        return "{}(key={}, factor={})".format(self.__class__.__name__, self.key, self.factor)


class ShiftProperty(BaseTransform):
    r"""Shifts the properties of a graph by a given factor.
    Args:
        keys (list of str): The keys of the properties to shift.
        factor (float): The shift factor.
    """

    def __init__(self, key, factor):
        self.key = key
        self.factor = factor

    def __call__(self, data):
        data[self.key] += self.factor
        return data

    def __repr__(self):
        return "{}(key={}, factor={})".format(self.__class__.__name__, self.key, self.factor)
