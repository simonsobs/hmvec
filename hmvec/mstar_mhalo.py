import numpy as np

"""
Stellar mass - halo mass relation.

Implementation from Matt Johnson and Moritz Munchmeyer.
Model from Behroozi, Conroy, & Wechsler 2010 (1001.0015), Table 2.
"""


def Mstellar_halo(z, log10mhalo):
    # Function to compute the stellar mass from a halo mass at redshift z.
    # z = list of redshifts
    # log10mhalo = log of the halo mass
    # FIXME: can the for loop be removed?
    # FIXME: is the zero indexing safe?

    log10mstar = np.linspace(-18, 18, 4000)[None, :]
    mh = Mhalo_stellar(z, log10mstar)
    mstar = np.zeros((z.shape[0], log10mhalo.shape[-1]))
    for i in range(z.size):
        mstar[i] = np.interp(log10mhalo[0], mh[i], log10mstar[0])
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
    log10M1 = M1 + M1a * (a - 1)
    log10Mstar0 = Mstar00 + Mstara * (a - 1)
    beta = beta0 + beta_a * (a - 1)
    gamma = gamma0 + gamma_a * (a - 1)
    delta = delta0 + delta_a * (a - 1)
    log10mstar = log10mstellar
    log10mh = (
        -0.5
        + log10M1
        + beta * (log10mstar - log10Mstar0)
        + 10 ** (delta * (log10mstar - log10Mstar0))
        / (1.0 + 10 ** (-gamma * (log10mstar - log10Mstar0)))
    )
    return log10mh


def Mhalo_stellar(z, log10mstellar):
    # Function to compute halo mass as a function of stellar mass.
    # z = list of redshifts
    # log10mhalo = log of the halo mass

    output = np.zeros((z.size, log10mstellar.shape[-1]))

    a = 1.0 / (1 + z)
    log10mstellar = log10mstellar + z * 0

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

    sel1 = np.where(z.reshape(-1) <= 0.8)
    output[sel1] = Mhalo_stellar_core(
        log10mstellar[sel1],
        a[sel1],
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

    sel1 = np.where(z.reshape(-1) > 0.8)
    output[sel1] = Mhalo_stellar_core(
        log10mstellar[sel1],
        a[sel1],
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
