import numpy as np
import astropy.cosmology as cosmo
import astropy.units as u
import astropy.constants as const


class Virial():
    def __init__(self, cosmology=cosmo.Planck13, redshift=0, overdensity=200):
        self.cosmology = cosmology
        self.redshift = redshift
        self.hubble = self.cosmology.H(redshift)
        self.critical_density = (3 * self.hubble**2) / (8 * np.pi * const.G)
        self.overdensity = overdensity

    @u.quantity_input(R=u.kpc)
    def compute_M_from_R(self, R):
        return (4 / 3 * np.pi * R**3 * self.critical_density * self.overdensity).to(u.M_sun)
        
    @u.quantity_input(V=u.km/u.s)
    def compute_M_from_V(self, V):
        return (V**3 / (10 * const.G * self.hubble)).to(u.M_sun)

    @u.quantity_input(M=u.M_sun)
    def compute_R_from_M(self, M):
        return ((3 * M / (4 * np.pi * self.critical_density * self.overdensity))**(1/3)).to(u.kpc)

    @u.quantity_input(V=u.km/u.s)
    def compute_R_from_V(self, V):
        return (V / (10 * self.hubble)).to(u.kpc)

    @u.quantity_input(R=u.kpc)
    def compute_V_from_R(self, R):
        return (10 * self.hubble * R).to(u.km / u.s)

    @u.quantity_input(M=u.M_sun)
    def compute_V_from_M(self, M):
        return ((10 * const.G * self.hubble * M)**(1/3)).to(u.km / u.s)



