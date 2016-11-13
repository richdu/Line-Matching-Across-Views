import numpy as np


def __find_region(img1, img2, pt1, pt2, m, n):
    u1, v1 = pt1[0], pt1[1]
    u2, v2 = pt2[0], pt2[1]
    h1, w1 = np.shape(img1)
    h2, w2 = np.shape(img2)
    L = min(u1, u2, m)
    R = min(w1-u1-1, w2-u2-1, m)
    U = min(v1, v2, n)
    D = min(h1-v1-1, h2-v2-1, n)

    return L, R, U, D


def standard_correlation(img1, img2, pt1, pt2, m, n):
    """
    Computes correlation between two points
    :param img1: 3 channel image (R, G, B)
    :param img2: 3 channel image (R, G, B)
    :param pt1: tuple (x, y)
    :param pt2: tuple (x, y)
    :param m: width of rectangle
    :param n: height of rectangle
    :return: correlation score
    """
    L, R, U, D = __find_region(img1, img2, pt1, pt2, m, n)

    u1, v1 = pt1[0], pt1[1]
    u2, v2 = pt2[0], pt2[1]

    neighbourhood1 = np.float64(img1[v1-U:v1+D+1, u1-L:u1+R+1].ravel())
    neighbourhood2 = np.float64(img2[v2-U:v2+D+1, u2-L:u2+R+1].ravel())
    mu1 = np.mean(neighbourhood1)
    mu2 = np.mean(neighbourhood2)
    sig1 = np.std(neighbourhood1)
    sig2 = np.std(neighbourhood2)

    neighbourhood1 -= mu1
    neighbourhood2 -= mu2

    score = np.dot(neighbourhood1, np.transpose(neighbourhood2))
    score /= len(neighbourhood1)*sig1*sig2
    return score


# TODO: H-Correlation
