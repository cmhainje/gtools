import numpy as np
import astropy.cosmology as cosmo
import astropy.units as u
import astropy.constants as const

from .virial import Virial


class Profile():
    def __init__(self, rho_0, r_s):
        self.rho_0 = rho_0
        self.r_s = r_s
        
    def density(self, r):
        x = r / self.r_s
        return self.rho_0 / (x * (1 + x)**2)

    def enclosed_mass(self, r):
        x = r / self.r_s
        return 4 * np.pi * self.rho_0 * self.r_s**3 * (np.log1p(x) - x / (1 + x))


class Params():
    def __init__(self, cosmology=cosmo.Planck13, redshift=0, overdensity=200, **kwargs):
        # REQUIRES one Virial parameter, key: 'M_vir', 'r_vir', or 'v_vir'
        # - if concentration not provided, it will be calculated using the virial
        #   mass and the Dutton-Maccio mass-concentration relation
        self.cosmology = cosmology
        self.redshift = redshift
        self.overdensity = overdensity
        self.virial = Virial(cosmology=cosmology, redshift=redshift, overdensity=overdensity)
        
        self.M_vir, self.r_vir, self.v_vir = self.virial.read_and_compute_others(**kwargs)

        if 'c' in kwargs:
            self.c = kwargs['c']
        elif not self.r_s is None:
            self.c = self.r_vir / self.r_s
        else:
            # compute concentration using Dutton-Maccio mass-concentration relation
            if overdensity == 200:
                self.c = dutton_maccio_200(self.M_vir, redshift=self.redshift, cosmology=self.cosmology)
            else:
                self.c = dutton_maccio_vir(self.M_vir, redshift=self.redshift, cosmology=self.cosmology)

        self.r_s = self.compute_r_s(self.r_vir, self.c)
        self.rho_0 = self.compute_rho_0(self.M_vir, self.r_s, self.c)

    @staticmethod
    def compute_rho_0(M_vir, r_s, c):
        return M_vir / (4 * np.pi * r_s**3 * (np.log1p(c) - c / (1 + c)))

    @staticmethod
    def compute_r_s(r_vir, c):
        return r_vir / c


class NFW(Params):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.profile = Profile(self.rho_0, self.r_s)

    def density(self, r):
        return self.profile.density(r)

    def enclosed_mass(self, r):
        return self.profile.enclosed_mass(r)

    def virial_mass(self):
        return self.M_vir

    def virial_radius(self):
        return self.r_vir

    def virial_velocity(self):
        return self.v_vir

    def concentration(self):
        return self.c

    def scale_radius(self):
        return self.r_s


def dutton_maccio_200(M, redshift=0, cosmology=cosmo.Planck13):
    # dutton maccio 2014 https://arxiv.org/pdf/1402.7073.pdf
    # uses best fit for Planck cosmology, overdensity=200
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * redshift**(1.21))
    b = -0.101 + 0.026 * redshift
    h = cosmology.H0 / (100 * u.km/u.s/u.Mpc)
    _M = M / (1e12 * u.M_sun / h)
    return np.power(10, a + b * np.log10(_M))

def dutton_maccio_vir(M, redshift=0, cosmology=cosmo.Planck13):
    # dutton maccio 2014 https://arxiv.org/pdf/1402.7073.pdf
    # uses best fit for Planck cosmology, virial overdensity
    a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * redshift**(1.08))
    b = -0.097 + 0.024 * redshift
    h = cosmology.H0 / (100 * u.km/u.s/u.Mpc)
    _M = M / (1e12 * u.M_sun / h)
    return np.power(10, a + b * np.log10(_M))
