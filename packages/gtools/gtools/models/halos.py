import numpy as np
import astropy.units as u
from scipy.optimize import root_scalar
from astropy.cosmology import Planck13

from .virial import Virial


def find_virial_radius(halo, v=None):
    if v is None:
        v = Virial()

    rho_vir = (v.overdensity * v.critical_density).to(u.M_sun / u.kpc ** 3).value
    res = root_scalar(
        lambda r: halo.average_enclosed_density(r) - rho_vir,
        x0=halo.r_s,
        x1=100,
        bracket=[1e-5, 1e5],
    )
    if not res.converged:
        raise ValueError("Root finder failed to converge.")
    return res.root


class NFW:
    def __init__(self, rho_s, r_s):
        self.rho_s = rho_s
        self.r_s = r_s

    @staticmethod
    def f(x):
        return np.log(1 + x) - x / (1 + x)

    @staticmethod
    def cvir_from_c200(c200, cosmology=Planck13, redshift=0):
        v = Virial(cosmology=cosmology, redshift=redshift)
        q = 200 / v.overdensity

        def opt(cvir):
            return (NFW.f(c200) / (q * NFW.f(cvir))) ** (1 / 3) - c200 / cvir

        res = root_scalar(opt, x0=c200, x1=c200 + 10)
        if not res.converged:
            raise ValueError("Root finder failed to converge.")
        return res.root

    @staticmethod
    def c200_from_cvir(cvir, cosmology=Planck13, redshift=0):
        v = Virial(cosmology=cosmology, redshift=redshift)
        q = 200 / v.overdensity

        def opt(c200):
            return (NFW.f(c200) / (q * NFW.f(cvir))) ** (1 / 3) - c200 / cvir

        res = root_scalar(opt, x0=cvir, x1=max(cvir - 10, 1 if cvir != 1 else 2))
        if not res.converged:
            raise ValueError("Root finder failed to converge.")
        return res.root

    @staticmethod
    def from_Mvir_cvir(Mvir, cvir, cosmology=Planck13, redshift=0):
        vvir = Virial(cosmology=cosmology, redshift=redshift)
        rvir = vvir.compute_R_from_M(Mvir * u.M_sun).value
        rs = rvir / cvir
        rhos = Mvir / (4 * np.pi * rs ** 3 * NFW.f(cvir))
        return NFW(rhos, rs)

    @staticmethod
    def from_M200_c200(M200, c200, cosmology=Planck13, redshift=0):
        # compute r_s
        v200 = Virial(overdensity=200, cosmology=cosmology, redshift=redshift)
        r200 = v200.compute_R_from_M(M200 * u.M_sun).value
        rs = r200 / c200

        # compute M_vir, c_vir
        cvir = NFW.cvir_from_c200(c200, cosmology=cosmology, redshift=redshift)
        Mvir = M200 * NFW.f(cvir) / NFW.f(c200)

        # compute scale radius
        rhos = Mvir / (4 * np.pi * rs ** 3 * NFW.f(cvir))
        return NFW(rhos, rs)

    def density(self, r):
        x = r / self.r_s
        return self.rho_s / (x * (1 + x) ** 2)

    def enclosed_mass(self, r):
        x = r / self.r_s
        return 4 * np.pi * self.rho_s * self.r_s ** 3 * self.f(x)

    def average_enclosed_density(self, r):
        x = r / self.r_s
        return 3 * self.rho_s * self.f(x) / (x ** 3)

    def virial_radius(self, v=None):
        return find_virial_radius(self, v=v)

    def virial_mass(self, v=None):
        return self.enclosed_mass(self.virial_radius(v=v))

    def concentration(self, v=None):
        return self.virial_radius(v=v) / self.r_s

    def to_hernquist(self, v=None):
        # compute hernquist scale radius
        r_vir = self.virial_radius(v=v)
        x = r_vir / self.r_s
        a_H = self.r_s / ((1 / np.sqrt(2 * self.f(x))) - (1 / x))

        # compute hernquist total mass
        M_vir = self.enclosed_mass(r_vir)
        c_vir = r_vir / self.r_s
        M_H = M_vir * (a_H / self.r_s) ** 2 / (2 * self.f(c_vir))

        # compute hernquist scale density
        rho_s_H = M_H / (2 * np.pi * a_H ** 3)
        return Hernquist(rho_s_H, a_H)

    def to_galic(self, v=None):
        if v is None:
            v = Virial(overdensity=200)
        r_vir = self.virial_radius(v=v)
        V_vir = v.compute_V_from_R(r_vir * u.kpc).to(u.km / u.s).value
        c = self.concentration(v=v)
        hubble = v.hubble.to(1 / u.s).value
        return "\n".join(
            [
                f"HUBBLE  {hubble:.10}",
                f"V200    {V_vir:.10}",
                f"CC      {c:.10}",
            ]
        )


class Hernquist:
    def __init__(self, rho_s, r_s):
        self.rho_s = rho_s
        self.r_s = r_s

    @staticmethod
    def from_total_mass(M_H, r_s):
        rho_s = M_H / (2 * np.pi * r_s ** 3)
        return Hernquist(rho_s, r_s)

    def density(self, r):
        x = r / self.r_s
        return self.rho_s / (x * (1 + x) ** 3)

    def enclosed_mass(self, r):
        x = r / self.r_s
        return 2 * np.pi * self.rho_s * self.r_s ** 3 * x ** 2 / ((1 + x) ** 2)

    def total_mass(self):
        return 2 * np.pi * self.rho_s * self.r_s ** 3

    def average_enclosed_density(self, r):
        x = r / self.r_s
        return (3 / 2) * self.rho_s / (x * (1 + x) ** 2)

    def virial_radius(self, v=None):
        return find_virial_radius(self, v=v)

    def virial_mass(self, v=None):
        r_vir = self.virial_radius(v)
        return self.enclosed_mass(r_vir)

    def to_NFW(self, v=None):
        # compute NFW scale radius
        r_vir = self.virial_radius(v=v)

        def opt(r_s_NFW):
            x = r_vir / r_s_NFW
            return 1 / np.sqrt(2 * NFW.f(x)) - 1 / x - r_s_NFW / self.r_s

        res = root_scalar(opt, x0=self.r_s, x1=100)
        if not res.converged:
            raise ValueError("Root finder failed to converge.")
        r_s_NFW = res.root

        # compute NFW virial mass
        M_H = self.total_mass()
        c_vir = r_vir / r_s_NFW
        M_vir = 2 * NFW.f(c_vir) * M_H * ((r_s_NFW / self.r_s) ** 2)

        # compute NFW scale density
        rho_s_NFW = M_vir / (4 * np.pi * r_s_NFW ** 3 * NFW.f(c_vir))
        return NFW(rho_s_NFW, r_s_NFW)

    def to_galic(self, v=None):
        if v is None:
            v = Virial(overdensity=200)
        nfw = self.to_NFW(v=v)
        return nfw.to_galic(v=v)


def dutton_maccio(M, v=None):
    if v is None:
        v = Virial()
    z = v.redshift
    h = v.hubble.value / 100

    if v.overdensity == 200:
        b = -0.101 + 0.026 * z
        a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z ** 1.21)
    else:
        b = -0.097 + 0.024 * z
        a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * z ** 1.08)

    logc = a + b * np.log10(M / (1e12 * h))
    return 10 ** logc


def dutton_maccio_200(M, v=None):
    if v is None:
        v = Virial(overdensity=200)
    z = v.redshift
    h = v.hubble.value / 100
    b = -0.101 + 0.026 * z
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z ** 1.21)
    logc = a + b * np.log10(M / (1e12 * h))
    return 10 ** logc


def dutton_maccio_vir(M, v=None):
    if v is None:
        v = Virial()
    z = v.redshift
    h = v.hubble.value / 100
    b = -0.097 + 0.024 * z
    a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * z ** 1.08)
    logc = a + b * np.log10(M / (1e12 * h))
    return 10 ** logc
