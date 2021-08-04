import numpy as np
import h5py
import os
import glob
import warnings

import gtools as gt

class Reader():
    def __init__(self, dirname: str, particle_types: dict, times) -> None:
        self.dirname = dirname
        self.times = np.array(times)
        self._validate_particle_types(particle_types)
        self.part_types = particle_types
        self._open_snapshots()
        self._compute_part_type_masks()

    def _validate_particle_types(self) -> None:
        # part_types is of the following form
        # PartType1, number[, mass]
        # part_types['mw_dark']  = {'type': 'PartType1', 'number': 100000, 'mass': 200}
        # part_types['mw_star']  = {'type': 'PartType2', 'number':  10000, 'mass': 100}
        # part_types['sgr_dark'] = {'type': 'PartType1', 'number':   1000, 'mass':  50}
        # part_types['sgr_star'] = {'type': 'PartType2', 'number':    100, 'mass':  10}

        numbers, masses = [], []

        for k, v in self.particle_types:
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
        
    def _open_snapshots(self):
        # opens h5py Files for each snapshot
        filename_prefix = os.path.join(self.dirname, 'snapshot_')
        snapshot_names = glob.glob(f'{filename_prefix}*.hdf5')
        if len(snapshot_names) == 0:
            raise ValueError('No snapshots found in given directory.')

        to_index = lambda s: int(s.replace(filename_prefix, '')[:-5])
        snapshot_names.sort(key=to_index)

        self.snapshots = []
        for snap in snapshot_names:
            try:
                file = h5py.File(snap, 'r')

                for part_type, info in self.part_types.items():
                    masses = file[info['type']]['Masses'][()]
                    if 'mass' in info.keys():
                        mask = (masses == info['mass'])
                    else:
                        unique, counts = np.unique(masses, return_counts=True)
                        mass = unique[np.nonzero(counts == info['number'])[0][0]]
                        mask = (masses == mass)
                    snap_masks[part_type] = mask

            except:
                warnings.warn(f'There was a problem opening snapshot'
                              f'{to_index(snap)} at {snap}'
                               'Proceeding without this snapshot.')
                file = None
                snap_masks = None

            self.snapshots.append(file)

    def _compute_part_type_masks(self) -> None:
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
                mass = unique[np.nonzero(counts == info['number'])[0][0]]
                mask = (masses == mass)
            self.masks[part_type] = mask
    
    def get_snap(self, index):
        # returns the snapshot with the given index
        if index < 0 or index >= len(self.snapshots):
            raise ValueError('Index out of bounds.')

        if self.snapshots[index] is None:
            raise ValueError('No valid snapshot with this index.')

        return self.snapshots[index]

    def get_time(self, index):
        # gets the time corresponding to the given index
        return self.times[index]

    def get_mass(self, index, part_type):
        # gets the masses for the specified snapshot index and particle type
        snap = self.get_snap(index)
        mask = self.masks[index][part_type]
        type = self.part_types[part_type]['type']

        ids = snap[type]['ParticleIDs'][()]
        masses = snap[type]['Masses'][()]

        # sort positions by ID before applying mask
        sorted_idx = np.argsort(ids)
        masses = masses[sorted_idx,:]
        masses = masses[mask,:]
        return masses * 1e10

    def get_pos(self, index, part_type, transform_cyl=False, transform_sph=False):
        # gets the positions for the specified snapshot index and particle type
        snap = self.get_snap(index)
        mask = self.masks[index][part_type]
        type = self.part_types[part_type]['type']

        ids = snap[type]['ParticleIDs'][()]
        pos = snap[type]['Coordinates'][()]

        # sort positions by ID before applying mask
        sorted_idx = np.argsort(ids)
        pos = pos[sorted_idx,:]
        pos = pos[mask,:]

        if transform_cyl:
            pos = gt.pos_cart_to_cyl(pos)

        if transform_sph:
            pos = gt.pos_cart_to_sph(pos)
        
        return pos

    def get_vel(self, index, part_type, transform_cyl=False, transform_sph=False):
        # gets the velocities for the specified snapshot index and particle type
        snap = self.get_snap(index)
        mask = self.masks[index][part_type]
        type = self.part_types[part_type]['type']

        ids = snap[type]['ParticleIDs'][()]
        vel = snap[type]['Velocities'][()]

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

