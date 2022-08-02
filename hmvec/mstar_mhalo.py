import numpy as np

"""
Stellar mass - halo mass relation.

Implementation from Matt Johnson and Moritz Munchmeyer.
Model from Behroozi, Conroy, & Wechsler 2010 (1001.0015), Table 2.
"""


def Mstellar_halo(z, log10mhalo):
    """Compute M_*(M_halo) at a list of redshifts.

    Parameters
    ----------
    z : array_like
        Array of redshifts to compute at.
    log10mhalo : array_like
        Array of log10(M_halo / M_sun), at each z, packed as [z,log10m].

    Returns
    -------
    log10mstar : array_like
        Array of log10(M_* / M_sun), packed as [z,log10m].
    """
    # FIXME: can the for loop be removed?

    # Precompute M_halo(M_*), packed as [z,log10m], over a wide range of M_*
    log10mstar = np.tile(np.linspace(-18, 18, 4000), (len(z), 1))
    mh = Mhalo_stellar(z, log10mstar)

    # For each z, compute M_*(M_halo) using linear interpolation
    mstar = np.zeros_like(log10mhalo)
    for i in range(len(z)):
        mstar[i] = np.interp(log10mhalo[i], mh[i], log10mstar[0])

    return mstar


def Mhalo_stellar_core(
    log10mstellar,
    a,
    Mstar00,
    Mstara,
    M1,
    M1a,
    beta0,
    beta_a,
    gamma0,
    gamma_a,
    delta0,
    delta_a,
):
    """Compute M_halo(M_*) at a list of scale factors.

    Parameters
    ----------
    log10mstellar : array_like
        Array of log10(M_* / M_sun) at each z, packed as [z,log10m].
    a : array_like
        Array of scale factors to compute at.
    Mstar00, Mstara, M1, M1a, beta0, beta_a, gamma0, gamma_a, delta0, delta_a : float
        Parameters in fitting function.

    Returns
    -------
    log10mhalo : array_like
        Array of log10(M_halo / M_sun), packed as [z,log10m].
    """
    a_in = a[:, None]
    log10M1 = M1 + M1a * (a_in - 1)
    log10Mstar0 = Mstar00 + Mstara * (a_in - 1)
    beta = beta0 + beta_a * (a_in - 1)
    gamma = gamma0 + gamma_a * (a_in - 1)
    delta = delta0 + delta_a * (a_in - 1)

    log10mh = (
        -0.5
        + log10M1
        + beta * (log10mstellar - log10Mstar0)
        + 10 ** (delta * (log10mstellar - log10Mstar0))
        / (1.0 + 10 ** (-gamma * (log10mstellar - log10Mstar0)))
    )

    return log10mh


def Mhalo_stellar(z, log10mstellar):
    """Compute M_halo(M_*) at a list of redshifts.

    Parameters
    ----------
    z : array_like
        Array of redshifts to compute at.
    log10mstellar : array_like
        Array of log10(M_* / M_sun) at each z, packed as [z,log10m].

    Returns
    -------
    log10mhalo : array_like
        Array of log10(M_halo / M_sun), packed as [z,log10m].
    """
    _Z_BREAK = 0.8

    output = np.zeros_like(log10mstellar, dtype=np.float64)

    a = 1 / (1 + z)

    Mstar00 = 10.72
    Mstara = 0.55
    M1 = 12.35
    M1a = 0.28
    beta0 = 0.44
    beta_a = 0.18
    gamma0 = 1.56
    gamma_a = 2.51
    delta0 = 0.57
    delta_a = 0.17

    if np.any(z <= _Z_BREAK):
        output[z <= _Z_BREAK] = Mhalo_stellar_core(
            log10mstellar[z <= _Z_BREAK],
            a[z <= _Z_BREAK],
            Mstar00,
            Mstara,
            M1,
            M1a,
            beta0,
            beta_a,
            gamma0,
            gamma_a,
            delta0,
            delta_a,
        )

    Mstar00 = 11.09
    Mstara = 0.56
    M1 = 12.27
    M1a = -0.84
    beta0 = 0.65
    beta_a = 0.31
    gamma0 = 1.12
    gamma_a = -0.53
    delta0 = 0.56
    delta_a = -0.12

    if np.any(z > _Z_BREAK):
        output[z > _Z_BREAK] = Mhalo_stellar_core(
            log10mstellar[z > _Z_BREAK],
            a[z > _Z_BREAK],
            Mstar00,
            Mstara,
            M1,
            M1a,
            beta0,
            beta_a,
            gamma0,
            gamma_a,
            delta0,
            delta_a,
        )

    return output
