"""
Unit tests for limber_integral function using RectBivariateSpline.

Note: The comparison tests against the old (deprecated) implementation using
interp2d and dfitpack.bispeu have been removed because these APIs were
completely removed in SciPy 1.14.0 and numpy.trapz was removed in NumPy 2.0.
"""
import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline, interp1d


def limber_integral_new(ells, zs, ks, Pzks, gzs, Wz1s, Wz2s, hzs, chis):
    """New implementation using RectBivariateSpline"""
    hzs = np.array(hzs).reshape(-1)
    Wz1s = np.array(Wz1s).reshape(-1)
    Wz2s = np.array(Wz2s).reshape(-1)
    chis = np.array(chis).reshape(-1)

    prefactor = hzs * Wz1s * Wz2s / chis**2.
    zevals = gzs
    if zs.size > 1:
        # RectBivariateSpline expects (x, y, z) where z has shape (len(x), len(y))
        f = RectBivariateSpline(ks, zs, Pzks.T, kx=3, ky=3)
    else:
        f = interp1d(ks, Pzks[0], bounds_error=True)
    Cells = np.zeros(ells.shape)
    for i, ell in enumerate(ells):
        kevals = (ell + 0.5) / chis
        if zs.size > 1:
            # ev() evaluates at pairs of points (kevals[i], zevals[i])
            interpolated = f.ev(kevals, zevals)
        else:
            interpolated = f(kevals)
        if zevals.size == 1:
            Cells[i] = np.sum(interpolated * prefactor)
        else:
            Cells[i] = np.trapezoid(interpolated * prefactor, zevals)
    return Cells


class TestLimberIntegral:
    """Test suite for the new RectBivariateSpline-based implementation"""

    def setup_method(self):
        """Set up test fixtures with realistic cosmological data"""
        # Multipoles
        self.ells = np.array([100, 200, 500, 1000, 2000])

        # Redshifts for P(k,z)
        self.zs = np.linspace(0.1, 2.0, 20)

        # Wavenumbers
        self.ks = np.logspace(-3, 1, 100)

        # Mock power spectrum P(z,k) - simple power law model
        self.Pzks = np.outer(
            (1 + self.zs)**(-2),  # redshift evolution
            self.ks**(-2.5) * np.exp(-self.ks/10)  # k dependence
        ) * 1e4

        # Redshifts for weight functions
        self.gzs = np.linspace(0.1, 2.0, 50)

        # Weight functions (mock lensing kernels)
        self.Wz1s = np.exp(-((self.gzs - 0.8)/0.3)**2)
        self.Wz2s = np.exp(-((self.gzs - 1.0)/0.4)**2)

        # Hubble parameter in 1/Mpc (mock values)
        H0 = 70 / 3e5  # H0 in 1/Mpc
        self.hzs = H0 * np.sqrt(0.3 * (1 + self.gzs)**3 + 0.7)

        # Comoving distances (mock values in Mpc)
        self.chis = 3000 * self.gzs  # simplified linear relation

    def test_2d_interpolation_runs(self):
        """Test that 2D interpolation case runs without errors"""
        Cells = limber_integral_new(
            self.ells, self.zs, self.ks, self.Pzks,
            self.gzs, self.Wz1s, self.Wz2s, self.hzs, self.chis
        )
        assert Cells.shape == self.ells.shape
        assert np.all(np.isfinite(Cells))

    def test_1d_interpolation_runs(self):
        """Test that 1D interpolation case runs without errors"""
        zs_1d = np.array([1.0])
        Pzks_1d = self.Pzks[len(self.zs)//2:len(self.zs)//2+1, :]

        Cells = limber_integral_new(
            self.ells, zs_1d, self.ks, Pzks_1d,
            self.gzs, self.Wz1s, self.Wz2s, self.hzs, self.chis
        )
        assert Cells.shape == self.ells.shape
        assert np.all(np.isfinite(Cells))

    def test_single_redshift_evaluation(self):
        """Test with single redshift in gzs"""
        gzs_single = np.array([1.0])
        Wz1s_single = np.array([1.0])
        Wz2s_single = np.array([1.0])
        hzs_single = np.array([0.0003])
        chis_single = np.array([3000.0])

        Cells = limber_integral_new(
            self.ells, self.zs, self.ks, self.Pzks,
            gzs_single, Wz1s_single, Wz2s_single, hzs_single, chis_single
        )
        assert Cells.shape == self.ells.shape
        assert np.all(np.isfinite(Cells))

    def test_output_shape(self):
        """Test that output shape matches ells shape"""
        Cells = limber_integral_new(
            self.ells, self.zs, self.ks, self.Pzks,
            self.gzs, self.Wz1s, self.Wz2s, self.hzs, self.chis
        )
        assert Cells.shape == self.ells.shape

    def test_non_negative_output(self):
        """Test that output is non-negative for positive inputs"""
        Cells = limber_integral_new(
            self.ells, self.zs, self.ks, np.abs(self.Pzks),
            self.gzs, np.abs(self.Wz1s), np.abs(self.Wz2s),
            np.abs(self.hzs), np.abs(self.chis)
        )
        assert np.all(Cells >= 0)

    def test_ell_scaling(self):
        """Test that C(ell) decreases with increasing ell (typical behavior)"""
        Cells = limber_integral_new(
            self.ells, self.zs, self.ks, self.Pzks,
            self.gzs, self.Wz1s, self.Wz2s, self.hzs, self.chis
        )
        # For typical cosmological power spectra, C(ell) should generally decrease
        # with ell at high ell (damping tail behavior)
        # Check that at least the trend is decreasing for the last few ells
        assert Cells[-1] < Cells[0]

    def test_interpolation_consistency(self):
        """Test that RectBivariateSpline interpolates correctly at grid points"""
        # Create a simple power spectrum where we know the values
        f = RectBivariateSpline(self.ks, self.zs, self.Pzks.T, kx=3, ky=3)

        # Test that interpolation at grid points recovers original values
        for i, k in enumerate(self.ks[5:-5:10]):  # Skip edges where spline might be less accurate
            for j, z in enumerate(self.zs[2:-2:4]):
                interpolated = f.ev(np.array([k]), np.array([z]))[0]
                ki = list(self.ks).index(k)
                zi = list(self.zs).index(z)
                original = self.Pzks[zi, ki]
                np.testing.assert_allclose(interpolated, original, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
