import sys
sys.path.append(r'...\TT-IRLS')
import scikit_tt as scikit
from typing import List
from scikit_tt.data_driven.transform import Function
import numpy as np
from itertools import product

def target_positions(dims):
    R = dims[-1]
    front_dims = dims[:-1]
    ranges = [range(1, d + 1) for d in front_dims]
    cartesian_product = list(product(*ranges))
    target_positions = np.hstack([
        np.arange(1, R + 1).reshape(-1, 1),
        np.array(cartesian_product, dtype=int)
    ])
    return target_positions

def build_psi(x, phi, lam, beta, eps):

    d, m = x.shape
    p = len(phi)
    n = p ** d

    # J
    result = target_positions([p] * d + [n])

    # initial cores
    cores = [np.zeros([1, m + n, 1, m + n])] + \
            [np.zeros([m + n, p, 1, m + n]) for _ in range(1, d)] + \
            [np.zeros([m + n, p, 1, 1])]
    # the first core
    cores[0] = np.eye(m + n).reshape(1, m + n, 1, m + n)
    # 2-dth cores
    for i in range(1, d):
        for j in range(m):
            cores[i][j, :, 0, j] = np.array([phi[k](x[i - 1, j]) for k in range(p)])
    cores[0]
    # 2-dth cores augmentation
    for i in range(1, d):
        for j in range(n):
            cores[i][m + j, result[j][i] - 1, 0, m + j] = 1

    # the last core
    for j in range(m):
        cores[d][j, :, 0, 0] = np.array([phi[k](x[d - 1, j]) for k in range(p)])
    # the last core augmentaion
    for j in range(n):
        cores[d][m + j, result[j][d] - 1, 0, 0] = lam*(beta[j]**2+eps**2)**(-1/4)

    psi = scikit.TT(cores)

    return psi

def coefficient_solving(x, psi, P, v, threshold):


    d, m = x.shape
    p = len(P)
    n = p ** d
    # v augmentation
    y = np.concatenate([v, np.zeros((1, n), dtype=v.dtype)], axis=1)
    # compute C
    xi = psi.pinvtwo(d, threshold=threshold, ortho_l=False, ortho_r=True)
    reshape_shape = [m + n, xi.ranks[1]]
    transformed_core = y.dot(xi.cores[0].reshape(reshape_shape))
    xi.cores[0] = transformed_core.reshape(1, 1, 1, xi.ranks[1])
    xi.row_dims[0] = 1

    return xi


def FiniteDiff(u, dx, d):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3

    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing

    [cf. PDE-FIND]
    """

    n = u.size
    ux = np.zeros(n, dtype=np.float32)

    if d == 1:
        for i in range(1, n - 1):
            ux[i] = (u[i + 1] - u[i - 1]) / (2 * dx)

        ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
        ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
        return ux

    if d == 2:
        for i in range(1, n - 1):
            ux[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2

        ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
        ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx ** 2
        return ux

    if d == 3:
        for i in range(2, n - 2):
            ux[i] = (u[i + 2] / 2 - u[i + 1] + u[i - 1] - u[i - 2] / 2) / dx ** 3

        ux[0] = (-2.5 * u[0] + 9 * u[1] - 12 * u[2] + 7 * u[3] - 1.5 * u[4]) / dx ** 3
        ux[1] = (-2.5 * u[1] + 9 * u[2] - 12 * u[3] + 7 * u[4] - 1.5 * u[5]) / dx ** 3
        ux[n - 1] = (2.5 * u[n - 1] - 9 * u[n - 2] + 12 * u[n - 3] - 7 * u[n - 4] + 1.5 * u[n - 5]) / dx ** 3
        ux[n - 2] = (2.5 * u[n - 2] - 9 * u[n - 3] + 12 * u[n - 4] - 7 * u[n - 5] + 1.5 * u[n - 6]) / dx ** 3
        return ux

    if d > 3:
        return FiniteDiff(FiniteDiff(u, dx, 3), dx, d - 3)




def coordinate_major(x: np.ndarray, phi: List[Function]) -> 'TT':

    m = x.shape[1] # number of snapshots
    p = len(phi) # number of modes
    d = x.shape[0] # number of dimensions

    # define cores as list of empty arrays
    cores = [np.zeros([1, p, 1, m])] + [np.zeros([m, p, 1, m]) for _ in range(1, d)]

    # insert elements of first core
    for j in range(m):
        cores[0][0, :, 0, j] = np.array([phi[k](x[0, j]) for k in range(p)])

    # insert elements of subsequent cores
    for i in range(1, d):
        for j in range(m):
            cores[i][j, :, 0, j] = np.array([phi[k](x[i, j]) for k in range(p)])

    # append core containing unit vectors
    cores.append(np.eye(m).reshape(m, m, 1, 1))

    # construct tensor train
    psi = scikit.TT(cores)
    return psi


def mandy_cm(x: np.ndarray, y: np.ndarray, phi: List[Function], threshold: float=0.0):

    d = x.shape[0]
    m = x.shape[1]

    # construct transformed data tensor
    psi = coordinate_major(x, phi)

    # define xi as pseudoinverse of psi
    xi = psi.pinv(d, threshold=threshold, ortho_r=False)

    # multiply last core with y
    xi.cores[d] = (xi.cores[d].reshape([xi.ranks[d], m]).dot(y.transpose())).reshape(xi.ranks[d], 1, 1, 1)

    # set new row dimension
    xi.row_dims[d] = 1

    return xi



def diff_periodic(u,dx,axis,d):

    if d == 1:
        up2 = np.roll(u, -2, axis)
        up1 = np.roll(u, -1, axis)
        um1 = np.roll(u, 1, axis)
        um2 = np.roll(u, 2, axis)
        ux = (-up2 + 8*up1 - 8*um1 + um2) / (12*dx)

    if d == 2:
        up2 = np.roll(u, -2, axis)
        up1 = np.roll(u, -1, axis)
        um1 = np.roll(u, 1, axis)
        um2 = np.roll(u, 2, axis)
        ux = (-up2 + 16 * up1 - 30 * u + 16 * um1 - um2) / (12 * dx ** 2)

    if d == 3:
        up3 = np.roll(u, -3, axis)
        up2 = np.roll(u, -2, axis)
        up1 = np.roll(u, -1, axis)
        um1 = np.roll(u, 1, axis)
        um2 = np.roll(u, 2, axis)
        um3 = np.roll(u, 3, axis)
        ux = (up3 - 8 * up2 + 13 * up1 - 13 * um1 + 8 * um2 - um3) / (8 * dx ** 3)
    if d == 4:
        up2 = np.roll(u, -2, axis)
        up1 = np.roll(u, -1, axis)
        um1 = np.roll(u, 1, axis)
        um2 = np.roll(u, 2, axis)
        ux = (-up2 + 4 * up1 - 6 * u + 4 * um1 - um2) / (dx ** 4)

    return ux


def dudt(u, dt):

    ut = np.zeros_like(u)
    ut[0, ...] = (u[1, ...] - u[0, ...]) / dt
    ut[1:-1, ...] = (u[2:, ...] - u[:-2, ...]) / (2 * dt)
    ut[-1, ...] = (u[-1, ...] - u[-2, ...]) / dt
    return ut


def dudt2(u, dt):

    ut = np.zeros_like(u)
    ut[0, ...] = (-3*u[0, ...] + 4*u[1, ...] - u[2, ...]) / (2*dt)
    ut[1:-1, ...] = (u[2:, ...] - u[:-2, ...]) / (2 * dt)
    ut[-1, ...] = (u[-1, ...] - u[-2, ...]) / dt
    return ut



