import numpy as np
import h5py
import os
import glob
import warnings

import gtools as gt

class Reader():
    def __init__(self, dirname: str, particle_types: dict, times) -> None:
        """Instantiates the reader object.

        Parameters
        ----------
        dirname : str
            Name of the directory containing the GIZMO snapshots.
        particle_types : dict
            A dict describing the types of particles in the snapshots. The keys
            of the dict will be the names of the particle types; any names are
            allowed. The values of the dict must be dicts themselves; one
            associated with each particle type.  Every particle type's dict must
            contain the field 'type' containing the name of the GIZMO particle
            type associated with the particles.  (e.g. 'PartType2' for stars).
            It must also contain either the 'number' or the 'mass' of the
            particle type (or both). If there is more than one particle type
            with the same number, mass is required to disambiguate. If there is
            more than one particle type with the same mass, number is required
            to disambiguate. Behavior is undefined if more than one particle
            type has the same number and mass.
        times : array_like
            An array containing the times associated with each snapshot.
        """

        self.dirname = dirname
        self.times = np.array(times)
        self._validate_particle_types(particle_types)
        self.part_types = particle_types
        try:
            self._open_snapshots()
            self._compute_part_type_masks()
        except:
            pass

    def _validate_particle_types(self, particle_types) -> None:
        """Helper method to validate the value passed to the `particle_types`
        argument of the instantiator.

        Parameters
        ----------
        particle_types : dict
            A dict of the particle types. The keys of the dict will be the names
            of the particle types; any names are allowed. The values of the dict
            must be dicts themselves; one associated with each particle type.
            Every particle type's dict must contain the field 'type' containing
            the name of the GIZMO particle type associated with the particles.
            (e.g. 'PartType2' for stars). It must also contain either the
            'number' or the 'mass' of the particle type (or both). If there is
            more than one particle type with the same number, mass is required
            to disambiguate. If there is more than one particle type with the
            same mass, number is required to disambiguate. Behavior is undefined
            if more than one particle type has the same number and mass.

        Raises
        ------
        ValueError
            When the dict for any particle type does not have a 'type' key.
        ValueError
            When the dict for any particle type does not have both 'mass' and
            'number'.
        ValueError
            When the dict contains more than one particle type with same number
            and no mass information.
        ValueError
            When the dict contains more than one particle type with the same
            mass and no number information.
        """

        numbers, masses = [], []

        for k, v in particle_types.items():
            if not 'type' in v.keys():
                raise ValueError("'type' is a required key for every type.")

            if not 'mass' in v.keys() and not 'number' in v.keys():
                raise ValueError("One of 'mass' and 'number' is required.")

            if 'number' in v.keys():
                numbers.append(v['number'])
            if 'mass' in v.keys():
                masses.append(v['mass'])

        numbers = np.array(numbers)
        masses = np.array(masses)

        if len(masses) == 0:
            _, counts = np.unique(numbers, return_counts=True)
            if np.count_nonzero(counts > 1) > 0:
                raise ValueError('More than one particle type with the '
                                 'same number, requires disambiguation by '
                                 'mass.')

        if len(numbers) == 0:
            _, counts = np.unique(masses, return_counts=True)
            if np.count_nonzero(counts > 1) > 0:
                raise ValueError('More than one particle type with the '
                                 'same mass, requires disambiguation by '
                                 'number.')

    def _open_snapshots(self, prefix='snapshot_'):
        """Private method to populate the snapshots attribute.

        Opens h5py File objects for each snapshot. Warns the user if any are
        unable to be opened. The File objects are stored in the snapshots
        attribute.
        """

        # opens h5py Files for each snapshot
        filename_prefix = os.path.join(self.dirname, prefix)
        snapshot_names = glob.glob(f'{filename_prefix}*.hdf5')
        if len(snapshot_names) == 0:
            raise ValueError('No snapshots found in given directory.')

        to_index = lambda s: int(s.replace(filename_prefix, '')[:-5])
        snapshot_names.sort(key=to_index)

        self.snapshots = []
        for snap in snapshot_names:
            try:
                f = h5py.File(snap, 'r')

            except:
                warnings.warn(f'There was a problem opening snapshot '
                              f'{to_index(snap)} at {snap}. '
                               'Proceeding without this snapshot.')
                f = None

            self.snapshots.append(f)

    def _compute_part_type_masks(self) -> None:
        """Private method to populate the masks attribute.

        For the first snapshot, creates boolean masks that identify which
        particles correspond to each particle type. These masks correspond to
        the particles _sorted by their IDs_ so that one mask can be used for
        every snapshot.
        """

        # computes masks for each particle type
        first_snap = self.snapshots[0]
        self.masks = dict()
        for part_type, info in self.part_types.items():
            ids = first_snap[info['type']]['ParticleIDs'][()]
            sort_idx = np.argsort(ids)
            masses = first_snap[info['type']]['Masses'][()]
            masses = masses[sort_idx]
            if 'mass' in info.keys():
                mask = (masses == info['mass'])
            else:
                unique, counts = np.unique(masses, return_counts=True)
                mass = unique[np.nonzero(counts == info['number'])[0]]
                mask = (masses == mass)
            self.masks[part_type] = mask

    def get_snap(self, index):
        """Returns the h5py File object for the specified snapshot.

        Parameters
        ----------
        index : int
            The index of the snapshot.

        Returns
        -------
        h5py.File
            The h5py File object for the specified snapshot.

        Raises
        ------
        ValueError
            When the index is negative or larger than the number of snapshots
            that were found.
        ValueError
            When the index corresponds to a snapshot that the reader was unable
            to open.
        """

        # returns the snapshot with the given index
        if index < 0 or index >= len(self.snapshots):
            raise ValueError('Index out of bounds.')

        if self.snapshots[index] is None:
            raise ValueError('No valid snapshot with this index.')

        return self.snapshots[index]

    def get_time(self, index):
        """Returns the time corresponding to the specified snapshot.

        Parameters
        ----------
        index : int
            The index of the snapshot.

        Returns
        -------
        float
            The times corresponding to the given snapshot.
        """

        return self.times[index]

    def get_ids(self, index, part_type):
        snap = self.get_snap(index)
        mask = self.masks[part_type]
        ptype = self.part_types[part_type]['type']
        ids = snap[ptype]['ParticleIDs'][()]

        # sort IDs before applying mask
        sorted_idx = np.argsort(ids)
        ids = ids[sorted_idx]
        ids = ids[mask]
        return ids

    def get_mass(self, index, part_type):
        """Returns the masses of the specified particles in the specified
        snapshot.

        Parameters
        ----------
        index : int
            The index of the snapshot.
        part_type : str
            The key to the part_types dict specifying the desired particle type.

        Returns
        -------
        array_like
            The (N,) array containing the masses.
        """

        snap = self.get_snap(index)
        mask = self.masks[part_type]
        ptype = self.part_types[part_type]['type']

        ids = snap[ptype]['ParticleIDs'][()]
        masses = snap[ptype]['Masses'][()]

        # sort positions by ID before applying mask
        sorted_idx = np.argsort(ids)
        masses = masses[sorted_idx]
        masses = masses[mask]
        return masses * 1e10

    def get_pos(self, index, part_type, transform_cyl=False, transform_sph=False):
        """Returns the positions of the specified particles in the specified
        snapshot.

        Parameters
        ----------
        index : int
            The index of the snapshot.
        part_type : str
            The key to the part_types dict specifying the desired particle type.
        transform_cyl : bool, optional
            If True, transforms the positions into cylindrical coordinates
            before returning. Defaults to False.
        transform_sph : bool, optional
            If True, transforms the positions into spherical coordinates before
            returning. Defaults to False.

        Returns
        -------
        array_like
            The (N,3) array containing the positions in the chosen coordinate
            system.
        """

        snap = self.get_snap(index)
        mask = self.masks[part_type]
        ptype = self.part_types[part_type]['type']

        ids = snap[ptype]['ParticleIDs'][()]
        pos = snap[ptype]['Coordinates'][()]

        # sort positions by ID before applying mask
        sorted_idx = np.argsort(ids)
        pos = pos[sorted_idx,:]
        pos = pos[mask]

        if transform_cyl:
            pos = gt.pos_cart_to_cyl(pos)

        if transform_sph:
            pos = gt.pos_cart_to_sph(pos)

        return pos

    def get_vel(self, index, part_type, transform_cyl=False, transform_sph=False):
        """Returns the velocities of the specified particles in the specified
        snapshot.

        Parameters
        ----------
        index : int
            The index of the snapshot.
        part_type : str
            The key to the part_types dict specifying the desired particle type.
        transform_cyl : bool, optional
            If True, transforms the velocities into cylindrical coordinates
            before returning. Defaults to False.
        transform_sph : bool, optional
            If True, transforms the velocities into spherical coordinates before
            returning. Defaults to False.

        Returns
        -------
        array_like
            The (N,3) array containing the velocities in the chosen coordinate
            system.
        """

        snap = self.get_snap(index)
        mask = self.masks[part_type]
        ptype = self.part_types[part_type]['type']

        ids = snap[ptype]['ParticleIDs'][()]
        vel = snap[ptype]['Velocities'][()]

        # sort positions by ID before applying mask
        sorted_idx = np.argsort(ids)
        vel = vel[sorted_idx,:]
        vel = vel[mask,:]

        if transform_cyl:
            pos = self.get_pos(index, part_type)
            vel = gt.vel_cart_to_cyl(vel, pos)

        if transform_sph:
            pos = self.get_pos(index, part_type)
            vel = gt.vel_cart_to_sph(vel, pos)

        return vel

