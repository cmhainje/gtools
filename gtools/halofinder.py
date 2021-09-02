import numpy as np
import astropy.cosmology as cosmo
import astropy.units as u
import astropy.constants as const
import scipy.integrate
import scipy.interpolate

class HaloFinder():
    def __init__(self, masses, positions, velocities, 
        init_bound=None, cosmology=cosmo.Planck13, redshift=0, overdensity=200
    ) -> None:
        """
        Instantiates a HaloFinder object.

        The HaloFinder object provides methods for performing the unbinding
        procedure of the Amiga Halo Finder (AHF). 

        Parameters
        ----------
        masses : array_like
            Masses of the dark matter particles of the halo. Expected to have a
            mass-like unit from the astropy.units module.
        positions : array_like
            Positions of the dark matter particles of the halo. Expected to have
            a position-like unit from the astropy.units module.
        velocities : array_like
            Velocities of the dark matter particles of the halo. Expected to
            have a velocity-like unit from the astropy.units module.
        init_bound : array_like, optional
            A pre-existing boolean mask defining bound/unbound particles, to be
            used particularly in cases where a bound mask exists from a prior
            time step in simulation. Defaults to None.
        cosmology : astropy.cosmology.Cosmology, optional
            A Cosmology object, defining various cosmological parameters of
            relevance. Used to compute the Hubble parameter for virial quantity
            computations. Defaults to astropy.cosmology.Planck13.
        redshift : int, optional
            The redshift at which to compute the Hubble parameter with the given
            cosmology. Defaults to 0.
        overdensity : int, optional
            The overdensity which defines the virial radius. Defaults to 200.
        """
        self.mass = masses
        self.pos  = positions
        self.vel  = velocities
        self.sort = np.arange(len(self.mass), dtype=int)
        self.bound = np.ones_like(self.sort, dtype=bool) if (init_bound is None) else init_bound

        self.center_and_sort()

        self.cosmo = cosmology
        self.z = redshift
        self.hubble = self.cosmo.H(self.z)
        self.overdensity = overdensity

    def unsort_bound(self):
        """
        Re-orders the existing bound mask to match the particle order of the
        original masses, positions, and velocities.

        Returns
        -------
        array_like
            The bound mask, re-ordered for use with the input data.
        """
        unsort_bound = np.zeros_like(self.bound, dtype=bool)
        unsort_bound[self.sort[self.bound]] = True
        return unsort_bound

    def center_and_sort(self):
        """
        Computes the radius and velocity of the particles with respect to the
        center of mass and center of velocity of the halo. 

        This function updates and creates several class attributes. `self.sort`
        is updated to contain the argsort of the particles by radius.
        `self.bound` is updated such that its order matches the new sort order.
        `self.r`, `self.v`, and `self.M` are created and set to contain the
        particle radii, velocities, and masses in radius-sorted-order with
        respect to the computed COM and COV.
        """
        # reorder bound mask to match original order
        unsort_bound = self.unsort_bound()

        # compute COM & COV of bound particles
        com = np.average(self.pos[unsort_bound], axis=0)
        cov = np.average(self.vel[unsort_bound], axis=0)

        # compute radii, velocities w.r.t. COM & COV
        r = np.sqrt(np.sum((self.pos - com)**2, axis=1))
        v = np.sqrt(np.sum((self.vel - cov)**2, axis=1))

        # sort particles by radius
        self.sort = np.argsort(r)
        self.r = r[self.sort]
        self.v = v[self.sort]
        self.M = self.mass[self.sort]

        # reorder bound mask to match new sorted order
        self.bound = unsort_bound[self.sort]

    def compute_virial(self):
        """
        Computes the virial mass and radius of the halo.

        Returns
        -------
        number
            The virial radius.
        
        number
            The virial mass.
        """
        # M_encl = np.cumsum(self.M[self.bound])
        # coeff = 0.5 * self.hubble**2 / const.G * self.overdensity
        # distance = np.abs(M_encl / self.r[self.bound]**3 - coeff)
        # idx = np.argmin(distance)

        # r_vir = (self.r[self.bound])[idx]
        # M_vir = M_encl[idx]
        # return r_vir, M_vir

        M_vir = np.sum(self.M)
        r_vir = 1e3 * u.kpc
        return r_vir, M_vir

    def compute_integral(self):
        """
        A helper function to compute the integral over M(<r)/r^2 for all values
        of the radius. Returns an interpolating function over the integral with
        units of M_sun / kpc.

        Returns
        -------
        function
            A function which takes a single argument, `r`, and returns the value
            of the integral from 0 to `r` of M(<r')/r'^2 at that radius.
        """
        # M_encl = np.cumsum(self.M[self.bound])
        # integral = scipy.integrate.cumtrapz(M_encl / self.r[self.bound]**2)
        # bin_middles = 0.5 * ((self.r[self.bound])[1:] + (self.r[self.bound])[:-1])

        M_encl = np.cumsum(self.M)
        integral = scipy.integrate.cumtrapz(M_encl / self.r**2)
        bin_middles = 0.5 * ((self.r)[1:] + (self.r)[:-1])
        interp = scipy.interpolate.interp1d(bin_middles, integral, fill_value='extrapolate')
        return lambda r: interp(r) * u.M_sun / u.kpc

    def compute_phi_0(self, M_vir, r_vir, integral):
        """
        Computes the integration constant for $\phi(r)$.

        Parameters
        ----------
        M_vir : number
            The virial mass. Expected to have a mass-like unit.
        r_vir : number
            The virial radius. Expected have a distance-like unit.
        integral : function
            The output of the `compute_integral` function.

        Returns
        -------
        number
            The value of phi_0 in units of km^2/s^2.
        """
        return - (const.G * (M_vir / r_vir + integral(r_vir))).to(u.km**2 / u.s**2)

    def compute_phi(self, integral, phi_0):
        """
        Creates a function that returns the value of the gravitational potential
        at a given radius.

        Parameters
        ----------
        integral : function
            The output of the `compute_integral` method.
        phi_0 : number
            The value of phi_0.

        Returns
        -------
        function
            A function which takes one argument, `r`, and returns the value of
            the gravitational potential in km^2/s^2 at that radius.
        """
        return lambda r : (const.G * integral(r) + phi_0).to(u.km**2 / u.s**2)

    def unbinding_step(self, verbose=True):
        """
        Helper function which performs a single step of the unbinding algorithm.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints the number of particles cut on every iteration.

        Returns
        -------
        bool
            If True, then no updates were necessary and the unbinding procedure
            has stabilized. If False, then particles were stripped and another
            itertion is required.
        """
        n_init = np.count_nonzero(self.bound)

        self.center_and_sort()

        # compute virial mass and radius
        r_vir, M_vir = self.compute_virial()

        # compute virial mass and radius if not given
        # if r_vir is None or M_vir is None:
        #     r_vir, M_vir = self.compute_virial()

        # precompute integral over M(<r)/r^2
        integral = self.compute_integral()

        # compute phi_0
        phi_0 = self.compute_phi_0(M_vir, r_vir, integral)

        # calculate escape velocity
        phi = self.compute_phi(integral, phi_0)
        v_esc = np.sqrt(2 * np.abs(phi(self.r[self.bound])))

        # determine particles that are unbound
        escaping = self.v[self.bound] > v_esc
        self.bound[self.bound] &= ~escaping
        is_stable = np.count_nonzero(escaping) == 0

        if verbose:
            print(f'cut {n_init} down to {np.count_nonzero(self.bound)}')
        return is_stable or np.count_nonzero(self.bound) <= 1

    def unbind(self, verbose=False):
        """
        Performs the unbinding procedure until stability is reached.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints the number of particles cut on every iteration.
        """
        is_stable = self.unbinding_step(verbose=verbose)
        while not is_stable:
            is_stable = self.unbinding_step(verbose=verbose)


