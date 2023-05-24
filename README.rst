fasterscipydists
=================
fasterscipydists provides faster scipy.stats distributions.

Distributions implemented:

* uniform
* norm (Gaussian)
* expon (Exponential)
* dirac (Delta function)

Functions implemented:

* mean
* std
* ppf
* pdf
* logpdf
* cdf

Contributions are welcome! Please open issues or pull requests!

Usage
-----

This is a drop-in replacement, so instead of::

	import scipy.stats
	rv = scipy.stats.norm(1, 2)

you would do::

	import fasterscipydists
	rv = fasterscipydists.norm(1, 2)


See the scipy.stats documentation https://docs.scipy.org/doc/scipy/reference/stats.html
