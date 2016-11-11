import numpy as np


def skewSymmetricMatrix (a):
    return np.matrix([[0, -a[2], a[1]],
                      [a[2], 0, -a[0]],
                      [-a[1], a[0], 0]])


def computeFundamentalMatrix(p1, p2):
    p = np.mat(p1)
    pprime = np.mat(p2)
    pinv = p.getT()*np.linalg.inv(p*p.getT())
    pperp = (np.eye(4)-pinv*p)*np.mat('1; 1; 1; 1')
    skew = skewSymmetricMatrix(pprime*pperp)
    F = skew*pprime*pinv
    return F


p1 = np.asarray([[1, 0, 30, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
p2 = np.asarray([[10, 0, 0, 0], [0, 12, 0, 0], [0, 0, 3, 0]])
computeFundamentalMatrix(p1, p2)