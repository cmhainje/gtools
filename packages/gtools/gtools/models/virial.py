import numpy as np
import astropy.cosmology as cosmo
import astropy.units as u
import astropy.constants as const


class Virial:
    def __init__(self, cosmology=cosmo.Planck13, redshift=0, overdensity=None):
        """A class for handling computations with Virial quantities.

        Parameters
        ----------
        cosmology : astropy.cosmology.Cosmology, optional
            An astropy Cosmology object for defining various cosmological
            parameters. Used to compute the Hubble parameter at the specified
            redshift. Defaults to cosmo.Planck13.
        redshift : int, optional
            The redshift at which to compute the Hubble parameter in the given
            cosmology. Defaults to 0.
        overdensity : int, optional
            The overdensity defining the Virial object. Defaults to None, in
            which case it is computed using the Bryan & Norman (1998)
            prescription from the cosmology and redshift.
        """
        self.cosmology = cosmology
        self.redshift = redshift
        self.hubble = cosmology.H(redshift)
        # self.critical_density = (3 * self.hubble**2) / (8 * np.pi * const.G)
        self.critical_density = cosmology.critical_density(redshift)

        if overdensity is None:
            O_L = 1 - cosmology.Om(redshift)
            self.overdensity = 18 * np.pi ** 2 - 82 * O_L - 39 * O_L ** 2
        else:
            self.overdensity = overdensity

    @u.quantity_input(R=u.kpc)
    def compute_M_from_R(self, R):
        """Compute the virial mass given the virial radius.

        Parameters
        ----------
        R : Quantity ['length']
            The virial radius. Must be an astropy Quantity with units of length.

        Returns
        -------
        Quantity ['mass']
            The virial mass, in units of solar masses.

        Notes
        -----
        Computes the virial mass from the virial radius using the following
        relation.

        .. math:: M_{vir} = (4 \pi / 3) r_{vir}^3 \rho_c \Delta_c,

        for :math:`\rho_c` the critical density and :math:`\Delta_c` the
        overdensity.

        References
        ----------
        .. [1] https://arxiv.org/pdf/astro-ph/9508025.pdf
        """
        return (4 / 3 * np.pi * R ** 3 * self.critical_density * self.overdensity).to(
            u.M_sun
        )

    @u.quantity_input(V=u.km / u.s)
    def compute_M_from_V(self, V):
        """Compute the virial mass given the virial velocity.

        Parameters
        ----------
        V : Quantity ['velocity']
            The virial velocity. Must be an astropy Quantity with units of velocity.

        Returns
        -------
        Quantity ['mass']
            The virial mass, in units of solar masses.

        Notes
        -----
        Computes the virial mass from the virial velocity using the following
        relation.

        .. math:: M_{vir} = V_{vir} / (10 G H(z)),

        for :math:`H(z)` the Hubble parameter at redshift :math:`z`.

        References
        ----------
        .. [1] https://arxiv.org/pdf/astro-ph/9707093.pdf
        """
        return (V ** 3 / (10 * const.G * self.hubble)).to(u.M_sun)

    @u.quantity_input(M=u.M_sun)
    def compute_R_from_M(self, M):
        """Compute the virial radius given the virial mass.

        Parameters
        ----------
        M : Quantity ['mass']
            The virial mass. Must be an astropy Quantity with units of mass.

        Returns
        -------
        Quantity ['length']
            The virial radius, in units of kpc.

        Notes
        -----
        Computes the virial mass from the virial radius using the following
        relation.

        .. math:: r_{vir} = (3 M_{vir} / (4 \pi \rho_c \Delta_c))^{1/3}

        for :math:`\rho_c` the critical density and :math:`\Delta_c` the
        overdensity.

        References
        ----------
        .. [1] https://arxiv.org/pdf/astro-ph/9508025.pdf
        """
        return (
            (3 * M / (4 * np.pi * self.critical_density * self.overdensity)) ** (1 / 3)
        ).to(u.kpc)

    @u.quantity_input(V=u.km / u.s)
    def compute_R_from_V(self, V):
        """Compute the virial radius given the virial velocity.

        Parameters
        ----------
        V : Quantity ['velocity']
            The virial velocity. Must be an astropy Quantity with units of velocity.

        Returns
        -------
        Quantity ['length']
            The virial radius, in units of kpc.

        Notes
        -----
        Computes the virial radius from the virial velocity using the following
        relation.

        .. math:: r_{vir} = V_{vir} / (10 H(z)),

        for :math:`H(z)` the Hubble parameter at redshift :math:`z`.

        References
        ----------
        .. [1] https://arxiv.org/pdf/astro-ph/9707093.pdf
        """
        return (V / (10 * self.hubble)).to(u.kpc)

    @u.quantity_input(R=u.kpc)
    def compute_V_from_R(self, R):
        """Compute the virial velocity given the virial radius.

        Parameters
        ----------
        R : Quantity ['length']
            The virial radius. Must be an astropy Quantity with units of length.

        Returns
        -------
        Quantity ['velocity']
            The virial velocity, in units of km/s.

        Notes
        -----
        Computes the virial velocity from the virial radius using the following
        relation.

        .. math:: V_{vir} = 10 H(z) r_{vir}

        for :math:`H(z)` the Hubble parameter at redshift :math:`z`.

        References
        ----------
        .. [1] https://arxiv.org/pdf/astro-ph/9707093.pdf
        """
        return (10 * self.hubble * R).to(u.km / u.s)

    @u.quantity_input(M=u.M_sun)
    def compute_V_from_M(self, M):
        """Compute the virial velocity given the virial mass.

        Parameters
        ----------
        M : Quantity ['mass']
            The virial mass. Must be an astropy Quantity with units of mass.

        Returns
        -------
        Quantity ['velocity']
            The virial velocity, in units of km/s.

        Notes
        -----
        Computes the virial velocity from the virial mass using the following
        relation.

        .. math:: V_{vir} = 10 * G * H(z) M_{vir},

        for :math:`H(z)` the Hubble parameter at redshift :math:`z`.

        References
        ----------
        .. [1] https://arxiv.org/pdf/astro-ph/9707093.pdf
        """
        return ((10 * const.G * self.hubble * M) ** (1 / 3)).to(u.km / u.s)

    @u.quantity_input(M=u.M_sun)
    def compute_others_from_M(self, M):
        return self.compute_R_from_M(M), self.compute_V_from_M(M)

    @u.quantity_input(R=u.kpc)
    def compute_others_from_R(self, R):
        return self.compute_M_from_R(R), self.compute_V_from_R(R)

    @u.quantity_input(V=u.km / u.s)
    def compute_others_from_V(self, V):
        return self.compute_M_from_V(V), self.compute_R_from_V(V)

    def read_and_compute_others(self, M_vir=None, r_vir=None, v_vir=None, **kwargs):
        # there must be at least one virial parameter
        if not (M_vir or r_vir or v_vir):
            raise ValueError("One virial parameter must be given.")

        # if only one is specified, use virial relationships to compute others
        if not (r_vir or v_vir):
            r_vir, v_vir = self.compute_others_from_M(M_vir)
        elif not (M_vir or v_vir):
            M_vir, v_vir = self.compute_others_from_R(r_vir)
        elif not (M_vir or r_vir):
            M_vir, r_vir = self.compute_others_from_R(v_vir)

        # if two are specified, use v = sqrt(G M / r)
        elif not (v_vir):
            v_vir = np.sqrt(const.G * M_vir / r_vir)
        elif not (r_vir):
            r_vir = const.G * M_vir / v_vir ** 2
        elif not (M_vir):
            M_vir = r_vir * v_vir ** 2 / const.G

        # if all three are specified, do nothing

        return M_vir, r_vir, v_vir
