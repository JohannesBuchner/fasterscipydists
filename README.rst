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

Speed-up
--------

speed.py reports these numbers::

	norm.pdf       :  7.6x faster with scipy.stats=0.126s, this=0.015s
	norm.logpdf    : 10.8x faster with scipy.stats=0.120s, this=0.010s
	norm.cdf       : 10.3x faster with scipy.stats=0.087s, this=0.008s
	norm.ppf       : 15.9x faster with scipy.stats=0.149s, this=0.009s
	uniform.pdf    :  9.2x faster with scipy.stats=0.114s, this=0.011s
	uniform.logpdf :  8.2x faster with scipy.stats=0.127s, this=0.014s
	uniform.cdf    :  2.4x faster with scipy.stats=0.084s, this=0.025s
	uniform.ppf    : 26.7x faster with scipy.stats=0.144s, this=0.005s
	expon.pdf      : 15.3x faster with scipy.stats=0.110s, this=0.007s
	expon.logpdf   : 26.7x faster with scipy.stats=0.112s, this=0.004s
	expon.cdf      : 10.9x faster with scipy.stats=0.097s, this=0.008s
	expon.ppf      : 21.8x faster with scipy.stats=0.154s, this=0.007s

Tests
-----

Systematic verification against scipy.stats is done in test.py. Run with::

	pytest test.py
