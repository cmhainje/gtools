import numpy as np

def pos_cart_to_cyl(position_vectors):
    """Lifted from utilities/basic/coordinate.py"""
    positions_new = np.zeros(position_vectors.shape, dtype=position_vectors.dtype)
    # R = sqrt(x^2 + y^2)
    positions_new[:, 0] = np.sqrt(np.sum(position_vectors[:, [0, 1]] ** 2, 1))
    # phi = arctan(y / x)
    positions_new[:, 1] = np.arctan2(position_vectors[:, 1], position_vectors[:, 0])
    positions_new[:, 1][positions_new[:, 1] < 0] += 2 * np.pi  # convert to [0, 2 * pi)
    # Z = z
    positions_new[:, 2] = position_vectors[:, 2]
    return positions_new

def pos_cart_to_sph(position_vectors):
    """Lifted from utilities/basic/coordinate.py"""
    positions_new = np.zeros(position_vectors.shape, dtype=position_vectors.dtype)
    # r = sqrt(x^2 + y^2 + z^2)
    positions_new[:, 0] = np.sqrt(np.sum(position_vectors ** 2, 1))
    # theta = arccos(z / r)
    positions_new[:, 1] = np.arccos(position_vectors[:, 2] / positions_new[:, 0])
    # phi = arctan(y / x)
    positions_new[:, 2] = np.arctan2(position_vectors[:, 1], position_vectors[:, 0])
    positions_new[:, 2][positions_new[:, 2] < 0] += 2 * np.pi  # convert to [0, 2 * pi)
    return positions_new

def vel_cart_to_cyl(velocity_vectors, position_vectors):
    """Lifted from utilities/basic/coordinate.py"""
    velocities_new = np.zeros(velocity_vectors.shape, dtype=velocity_vectors.dtype)

    # convert position vectors
    # R = {x,y}
    R = position_vectors[:, [0, 1]]
    R_norm = np.zeros(R.shape, position_vectors.dtype)
    # R_total = sqrt(x^2 + y^2)
    R_total = np.sqrt(np.sum(R ** 2, 1))
    masks = np.where(R_total > 0)[0]
    # need to do this way
    R_norm[masks] = np.transpose(R[masks].transpose() / R_total[masks])

    # v_R = dot(v_{x,y}, R_norm)
    velocities_new[:, 0] = np.sum(velocity_vectors[:, [0, 1]] * R_norm, 1)
    # v_phi = cross(R_norm, v_{x,y})
    velocities_new[:, 1] = np.cross(R_norm, velocity_vectors[:, [0, 1]])
    # v_Z = v_z
    velocities_new[:, 2] = velocity_vectors[:, 2]
    return velocities_new

def vel_cart_to_sph(velocity_vectors, position_vectors):
    """Lifted from utilities/basic/coordinate.py"""
    velocities_new = np.zeros(velocity_vectors.shape, dtype=velocity_vectors.dtype)

    # convert position vectors
    # R = {x,y}
    R = position_vectors[:, [0, 1]]
    R_norm = np.zeros(R.shape, position_vectors.dtype)
    # R_total = sqrt(x^2 + y^2)
    R_total = np.sqrt(np.sum(R ** 2, 1))
    masks = np.where(R_total > 0)[0]
    # need to do this way
    R_norm[masks] = np.transpose(R[masks].transpose() / R_total[masks])

    # convert position vectors
    position_vectors_norm = np.zeros(position_vectors.shape, position_vectors.dtype)
    position_vectors_total = np.sqrt(np.sum(position_vectors ** 2, 1))
    masks = np.where(position_vectors_total > 0)[0]
    # need to do this way
    position_vectors_norm[masks] = np.transpose(
        position_vectors[masks].transpose() / position_vectors_total[masks]
    )

    # v_r = dot(v, r)
    velocities_new[:, 0] = np.sum(velocity_vectors * position_vectors_norm, 1)
    # v_theta
    a = np.transpose(
        [
            R_norm[:, 0] * position_vectors_norm[:, 2],
            R_norm[:, 1] * position_vectors_norm[:, 2],
            -R_total / position_vectors_total,
        ]
    )
    velocities_new[:, 1] = np.sum(velocity_vectors * a, 1)
    # v_phi = cross(R_norm, v_{x,y})
    velocities_new[:, 2] = np.cross(R_norm, velocity_vectors[:, [0, 1]])
    return velocities_new