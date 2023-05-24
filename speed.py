import scipy.stats
import fasterscipydists
import numpy as np
from timeit import timeit

a = np.random.uniform(size=100)

for dist in 'norm', 'uniform', 'expon':
	rv_1 = getattr(scipy.stats, dist)()
	rv_2 = getattr(fasterscipydists, dist)()
	for funcname in 'pdf', 'logpdf', 'cdf', 'ppf':
		func_1 = getattr(rv_1, funcname)
		t_1 = timeit(lambda : func_1(a), number=1000)
		func_2 = getattr(rv_2, funcname)
		t_2 = timeit(lambda : func_2(a), number=1000)
		print("%-15s: %5s faster with scipy.stats=%.3fs, this=%.3fs" % (
			dist + '.' + funcname, '%.1fx' % (t_1/t_2 - 1), t_1, t_2))
