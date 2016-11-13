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


def standard_correlation (img1, img2, pt1, pt2, m, n):
    L, R, U, D = __find_region(img1, img2, pt1, pt2, m, n)

    u1, v1 = pt1[0], pt1[1]
    u2, v2 = pt2[0], pt2[1]

    nbhd1 = np.float64(img1[v1-U:v1+D+1, u1-L:u1+R+1].ravel())
    nbhd2 = np.float64(img2[v2-U:v2+D+1, u2-L:u2+R+1].ravel())
    mu1 = np.mean(nbhd1); sig1 = np.std(nbhd1)
    mu2 = np.mean(nbhd2); sig2 = np.std(nbhd2)

    nbhd1 -= mu1
    nbhd2 -= mu2

    score = np.dot(nbhd1, np.transpose(nbhd2))
    score /= len(nbhd1)*sig1*sig2
    return score


# TODO: H-Correlation
