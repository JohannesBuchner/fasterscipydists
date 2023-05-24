import scipy
import numpy as np

class DeltaDist(object):
    """Dirac Delta distribution."""

    # provides compatibility with scipy.stats distributions when a parameter is fixed
    def __init__(self, value):
        self.value = value

    def ppf(self, u):
        return self.value

    def mean(self):
        return self.value

    def std(self):
        return 0

class ExponentialDist(object):
    """Faster exponential distribution than the scipy implementation."""

    # provides compatibility with scipy.stats distributions when a parameter is fixed
    def __init__(self, scale):
        self.scale = scale

    def ppf(self, u):
        return self.scale * -np.log1p(-u)

    def mean(self):
        return self.scale

    def std(self):
        return self.scale

class NormalDist(object):
    """Faster Gaussian distribution than the scipy implementation."""

    # provides compatibility with scipy.stats distributions when a parameter is fixed
    def __init__(self, mean, scale):
        self.scale = scale
        self.mean = mean

    def ppf(self, u):
        return scipy.special.ndtri(u) * self.scale + self.mean

    def mean(self):
        return self.mean

    def std(self):
        return self.std
