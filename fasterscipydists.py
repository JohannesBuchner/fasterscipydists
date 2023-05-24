"""
Fast probability distribution functions
---------------------------------------

This is a drop-in replacement for some scipy.stats distributions, so instead of::

	import scipy.stats
	rv = scipy.stats.norm(1, 2)

you would do::

	import fasterscipydists
	rv = fasterscipydists.norm(1, 2)

See the scipy.stats documentation https://docs.scipy.org/doc/scipy/reference/stats.html

License
-------

Copyright (c) 2023 Johannes Buchner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
