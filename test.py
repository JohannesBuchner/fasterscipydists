import fasterscipydists as stats
import scipy.stats
import numpy as np

def test_expon():
	rng = np.random.RandomState(41)
	for i in range(10):
		scale = 1. / rng.uniform()
		print(scale)
		ref = scipy.stats.expon(scale=scale)
		rv = stats.expon(scale=scale)
		np.testing.assert_allclose(ref.mean(), rv.mean())
		np.testing.assert_allclose(ref.std(), rv.std())
		x = 10**rng.uniform(-6, 6)
		np.testing.assert_allclose(ref.pdf(x), rv.pdf(x))
		np.testing.assert_allclose(ref.logpdf(x), rv.logpdf(x))
		np.testing.assert_allclose(ref.cdf(x), rv.cdf(x))
		us = rng.uniform(size=100)
		np.testing.assert_allclose(ref.ppf(us), rv.ppf(us))


def test_norm_vectorized():
	rng = np.random.RandomState(42)
	loc = np.array([0, 1, 10, 10000])
	scale = np.array([1e-3, 2, 100, 10])
	ref = scipy.stats.norm(loc=loc, scale=scale)
	rv = stats.norm(loc=loc, scale=scale)
	np.testing.assert_allclose(rv.mean(), loc)
	np.testing.assert_allclose(rv.std(), scale)
	np.testing.assert_allclose(ref.mean(), rv.mean())
	np.testing.assert_allclose(ref.std(), rv.std())
	x = 10**rng.uniform(-6 + loc, 6 + loc)
	np.testing.assert_allclose(ref.pdf(x), rv.pdf(x))
	xs = 10**rng.uniform(-6 + loc, 6 + loc, size=(100, len(loc)))
	np.testing.assert_allclose(ref.pdf(xs), rv.pdf(xs))
	np.testing.assert_allclose(ref.logpdf(x), rv.logpdf(x))
	np.testing.assert_allclose(ref.logpdf(xs), rv.logpdf(xs))
	np.testing.assert_allclose(ref.cdf(x), rv.cdf(x))
	np.testing.assert_allclose(ref.cdf(xs), rv.cdf(xs))
	us = rng.uniform(size=(100, len(loc)))
	np.testing.assert_allclose(ref.ppf(us), rv.ppf(us))

def test_norm():
	rng = np.random.RandomState(42)
	for i in range(10):
		scale = rng.uniform(0, 1)
		loc = rng.uniform(-10000, 10000)
		print(loc, scale)
		ref = scipy.stats.norm(loc=loc, scale=scale)
		rv = stats.norm(loc=loc, scale=scale)
		assert rv.mean() == loc
		assert rv.std() == scale
		np.testing.assert_allclose(ref.mean(), rv.mean())
		np.testing.assert_allclose(ref.std(), rv.std())
		x = 10**rng.uniform(-6, 6)
		np.testing.assert_allclose(ref.pdf(x), rv.pdf(x))
		np.testing.assert_allclose(ref.logpdf(x), rv.logpdf(x))
		np.testing.assert_allclose(ref.cdf(x), rv.cdf(x))
		xs = rng.uniform(loc - 10 * scale, loc + 10 * scale, size=100)
		np.testing.assert_allclose(ref.pdf(xs), rv.pdf(xs))
		np.testing.assert_allclose(ref.logpdf(xs), rv.logpdf(xs))
		np.testing.assert_allclose(ref.cdf(xs), rv.cdf(xs))
		us = rng.uniform(size=100)
		np.testing.assert_allclose(ref.ppf(us), rv.ppf(us))


def test_uniform_vectorized():
	rng = np.random.RandomState(42)
	loc = np.array([0, 1, 10, 10000])
	scale = np.array([1e-3, 2, 100, 10])
	ref = scipy.stats.uniform(loc=loc, scale=scale)
	rv = stats.uniform(loc=loc, scale=scale)
	np.testing.assert_allclose(ref.mean(), rv.mean())
	np.testing.assert_allclose(ref.std(), rv.std())
	x = 10**rng.uniform(-6 + loc, 6 + loc)
	np.testing.assert_allclose(ref.pdf(x), rv.pdf(x))
	xs = 10**rng.uniform(-6 + loc, 6 + loc, size=(100, len(loc)))
	np.testing.assert_allclose(ref.pdf(xs), rv.pdf(xs))
	np.testing.assert_allclose(ref.logpdf(x), rv.logpdf(x))
	np.testing.assert_allclose(ref.logpdf(xs), rv.logpdf(xs))
	np.testing.assert_allclose(ref.cdf(x), rv.cdf(x))
	np.testing.assert_allclose(ref.cdf(xs), rv.cdf(xs))
	us = rng.uniform(size=(100, len(loc)))
	np.testing.assert_allclose(ref.ppf(us), rv.ppf(us))

def test_uniform():
	rng = np.random.RandomState(42)
	for i in range(10):
		scale = rng.uniform(0, 1)
		loc = rng.uniform(-10000, 10000)
		print(loc, scale)
		ref = scipy.stats.uniform(loc=loc, scale=scale)
		rv = stats.uniform(loc=loc, scale=scale)
		np.testing.assert_allclose(ref.mean(), rv.mean())
		np.testing.assert_allclose(ref.std(), rv.std())
		x = 10**rng.uniform(-6, 6)
		np.testing.assert_allclose(ref.pdf(x), rv.pdf(x))
		np.testing.assert_allclose(ref.logpdf(x), rv.logpdf(x))
		np.testing.assert_allclose(ref.cdf(x), rv.cdf(x))
		xs = rng.uniform(loc - 10 * scale, loc + 10 * scale, size=100)
		np.testing.assert_allclose(ref.pdf(xs), rv.pdf(xs))
		np.testing.assert_allclose(ref.logpdf(xs), rv.logpdf(xs))
		np.testing.assert_allclose(ref.cdf(xs), rv.cdf(xs))
		us = rng.uniform(size=100)
		np.testing.assert_allclose(ref.ppf(us), rv.ppf(us))