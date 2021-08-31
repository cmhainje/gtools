from astropy.units.cgs import C
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
        return self.rho_0 / (x * (1 + x)**3)

    def enclosed_mass(self, r):
        x = r / self.r_s
        return 2 * np.pi * self.rho_0 * self.r_s**3 * x**2 / (1 + x)**2


class Params():
    def __init__(self, cosmology=cosmo.Planck13, redshift=0, overdensity=200, **kwargs):
        # possible combinations of inputs
        # - Virial parameter + NFW concentration
        # - Virial parameter + scale radius
        # - total mass + scale radius
        self.cosmology = cosmology 
        self.redshift = redshift 
        self.overdensity = overdensity 
        self.virial = Virial(cosmology=cosmology, redshift=redshift, overdensity=overdensity)
        
        if any(key in kwargs for key in ['M_vir', 'r_vir', 'v_vir']):
            self.M_vir, self.r_vir, self.v_vir = self.virial.read_and_compute_others(kwargs)
            if 'c' in kwargs:
                c = kwargs['c']
                nfw_r_s = self.r_vir / c
                self.r_s = nfw_r_s * np.sqrt( 2 * (np.log1p(c) - c/(1+c)) )
            elif 'r_s' in kwargs:
                self.r_s = kwargs['r_s']
            else:
                raise ValueError('One of {NFW concentration, Hernquist scale radius} required '
                                 'when specifying Virial mass.')
            self.rho_0 = self.compute_rho_0_from_virial()
            self.M_tot = 2 * np.pi * self.rho_0 * self.r_s**3

        elif 'M_tot' in kwargs:
            self.M_tot = kwargs['M_tot']
            if 'r_s' in kwargs:
                self.r_s = kwargs['r_s']
            else:
                raise ValueError('Scale radius required when specifying total mass.')
            self.rho_0 = self.compute_rho_0_from_total()
            self.r_vir = self.compute_virial_radius()
            self.M_vir, self.v_vir = self.virial.compute_others_from_R(self.r_vir)

        else:
            raise ValueError('Must specify mass, either Virial or total.')

    def compute_rho_0_from_total(self):
        return self.M_tot / (2 * np.pi * self.r_s**3)

    def compute_rho_0_from_virial(self):
        r_vir = self.virial.compute_R_from_M(self.M_vir * u.M_sun).value
        x = r_vir / self.r_s
        return self.M_vir / (2 * np.pi * self.r_s**3 * x**2 / (1 + x)**2)

    def compute_virial_radius(self):
        from scipy.optimize import root_scalar
        prof = Profile(self.rho_0, self.r_s)
        rho_c = self.cosmology.critical_density(self.redshift)
        target = (rho_c * self.overdensity).value
        return root_scalar( lambda r: prof.density(r).value - target ).root * u.kpc


class Hernquist(Params):
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

    def total_mass(self):
        return self.M_tot

    def scale_radius(self):
        return self.r_s


