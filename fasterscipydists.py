
import scipy
import numpy as np

class delta(object):
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


class uniform(object):
    """Uniform distribution."""

    # provides compatibility with scipy.stats distributions when a parameter is fixed
    def __init__(self, loc, scale):
        self.lo = loc
        self.hi = loc + scale
        self.scale = scale
        self.mid = loc + scale * 0.5

    def ppf(self, u):
        return self.scale * u + self.lo

    def cdf(self, x):
        return np.clip((x - self.lo) / self.scale, 0, 1)

    def pdf(self, x):
        return np.where(np.logical_and(x > self.lo, x < self.hi), 1. / self.scale, 0)

    def logpdf(self, x):
        return np.where(np.logical_and(x > self.lo, x < self.hi), -np.log(self.scale), -np.inf)

    def mean(self):
        return self.mid

    def std(self):
        return 12**-0.5 * self.scale


class expon(object):
    """Faster exponential distribution than the scipy implementation."""

    # provides compatibility with scipy.stats distributions when a parameter is fixed
    def __init__(self, scale):
        self.scale = scale
        self.lam = 1. / scale
        self.loglam = np.log(self.lam)

    def ppf(self, u):
        return self.scale * -np.log1p(-u)

    def pdf(self, x):
        return self.lam * np.exp(-self.lam * x)

    def logpdf(self, x):
        return self.loglam - self.lam * x

    def cdf(self, x):
        return 1 - np.exp(-self.lam * x)

    def mean(self):
        return self.scale

    def std(self):
        return self.scale


class norm(object):
    """Faster Gaussian distribution than the scipy implementation."""

    # provides compatibility with scipy.stats distributions when a parameter is fixed
    def __init__(self, loc, scale):
        assert np.all(scale > 0)
        self.scale = scale
        self.loc = loc
        self.C = (2 * np.pi * self.scale**2)**-0.5
        self.lnC =  -0.5 * np.log(2 * np.pi * self.scale**2)

    def ppf(self, u):
        return scipy.special.ndtri(u) * self.scale + self.loc

    def pdf(self, x):
        z = (x - self.loc) / self.scale
        return np.exp(-0.5 * z**2) * self.C

    def cdf(self, x):
        z = (x - self.loc) / self.scale
        return scipy.special.ndtr(z)

    def logpdf(self, x):
        z = (x - self.loc) / self.scale
        return -0.5 * z**2 + self.lnC

    def mean(self):
        return self.loc

    def std(self):
        return self.scale
