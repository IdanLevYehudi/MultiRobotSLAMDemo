import numpy as np


def wrap_to_pi(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def rotation_mat_2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s],
                     [s, c]])


def are_clockwise(v1, v2):
    """
    :param v1: 2xN array of vectors
    :param v2: 2xN array of vectors
    :return: True if v2 is in clockwise orientation relative to v1.
    """
    return -v1[0] * v2[1] + v1[1] * v2[0] > 0


def is_within_radius(v, r_squared):
    # v is of size 2xN
    return np.sum(np.square(v), axis=0) <= r_squared


def robot_to_global_points(robot_xytheta, points):
    """
    :param robot_xytheta: np array of [x, y, theta]
    :param points: 2xN array of points [[x],[y]] format
    :return: 2xN array of points in global reference frame
    """
    rot_mat = rotation_mat_2d(robot_xytheta[2])
    translation = robot_xytheta[:2, np.newaxis]
    return np.matmul(rot_mat, points.reshape((2, -1))) + translation


def global_to_robot_points(robot_xytheta, points):
    """
    :param robot_xytheta: np array of [x, y, theta]
    :param points: 2xN array of points [[x],[y]] format
    :return: 2xN array of points in robot's reference frame
    """
    rot_mat = rotation_mat_2d(robot_xytheta[2])
    translation = robot_xytheta[:2, np.newaxis]
    return np.matmul(rot_mat.T, points.reshape((2, -1)) - translation)


def robot_to_global_poses(robot_xytheta, poses):
    """
    :param robot_xytheta: np array of [x, y, theta]
    :param poses: 3xN array of poses [[x],[y],[theta]] format
    :return: 3xN array of poses in global reference frame
    """
    return np.vstack([robot_to_global_points(robot_xytheta, poses[:2]),
                      wrap_to_pi(poses[2] + robot_xytheta[2])])


def global_to_robot_poses(robot_xytheta, poses):
    """
    :param robot_xytheta: np array of [x, y, theta]
    :param poses: 3xN array of poses [[x],[y],[theta]] format
    :return: 3xN array of poses in robot's reference frame
    """
    return np.vstack([global_to_robot_points(robot_xytheta, poses[:2]),
                      wrap_to_pi(poses[2] - robot_xytheta[2])])


def xy_to_rangetheta(xy):
    """
    Converts 2D points from cartesian to polar representation.
    :param xy: numpy array of 2xN
    :return: 2xN array of [[range],[theta]]
    """
    range = np.linalg.norm(xy, axis=0)
    theta = np.arctan2(xy[1], xy[0])
    return np.vstack([range, theta])


def rangetheta_to_xy(rangetheta):
    """
    Converts 2D points from polar representation to cartesian.
    :param rangetheta: numpy array of 2xN
    :return: 2xN array of [[x],[y]]
    """
    return rangetheta[0] * np.vstack([np.cos(rangetheta[1]), np.sin(rangetheta[1])])


def intersect_objects_ids(objects_seen_1, objects_seen_2):
    """
    Returns the ids of objects present in both objects_seen arrays.
    :param objects_seen_1: 3xN array of objects seen by a robot.
    :param objects_seen_2: 3xN array of objects seen by a robot.
    :return: 1d np array.
    """
    return np.intersect1d(objects_seen_1[2], objects_seen_2[2])
