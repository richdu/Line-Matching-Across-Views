import numpy as np


def skew_symmetric_matrix (a):
    return np.matrix([[0, -a[2], a[1]],
                      [a[2], 0, -a[0]],
                      [-a[1], a[0], 0]])


def compute_fundamental(p1, p2):
    p = np.mat(p1)
    pprime = np.mat(p2)
    pinv = p.getT()*np.linalg.inv(p*p.getT())
    pperp = (np.eye(4)-pinv*p)*np.mat('1; 1; 1; 1')
    skew = skew_symmetric_matrix(pprime * pperp)
    F = skew*pprime*pinv
    return F