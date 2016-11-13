import numpy as np


def skew_symmetric_matrix(a):
    """
    Returns skew matrix that represents a cross product
    :param a: 3D vector
    :return: 3x3 skew matrix
    """
    return np.matrix([[0, -a[2], a[1]],
                      [a[2], 0, -a[0]],
                      [-a[1], a[0], 0]])


def compute_fundamental(p1, p2):
    """
    Computes fundamental matrix given 2 camera matrices
    :param p1: 3x4 camera projection matrix
    :param p2: 3x4 camera projection matrix
    :return: 3x3 fundamental matrix
    """
    p = np.mat(p1)
    p_prime = np.mat(p2)
    p_inv = np.linalg.pinv(p)
    p_null = (np.eye(4)-p_inv*p)*np.mat('1; 1; 1; 1')
    skew = skew_symmetric_matrix(p_prime * p_null)
    F = skew*p_prime*p_inv
    return F
