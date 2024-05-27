def mandy_cm(x: np.ndarray, y: np.ndarray, phi: List[Function], p: List, threshold: float=0.0):
    d = x.shape[0]
    m = x.shape[1]

    # construct transformed data tensor
    psi = coordinate_major(x, phi, p)

    # define xi as pseudoinverse of psi
    xi = psi.pinv(d, threshold=threshold, ortho_r=False)

    # multiply last core with y
    xi.cores[d] = (xi.cores[d].reshape([xi.ranks[d], m])@y.T).reshape(xi.ranks[d], y.shape[0], 1, 1)

    # set new row dimension
    xi.row_dims[d] = 1

    return xi



def coordinate_major(x: np.ndarray, phi: List[Function], p: List) -> 'TT':
    d = x.shape[0]
    m = x.shape[1]

    # define cores as list of empty arrays
    cores = [np.zeros([1, p[0], 1, m])] + [np.zeros([m, p[i], 1, m]) for i in range(1, d)]

    # insert elements of first core
    for j in range(m):
        cores[0][0, :, 0, j] = np.array([phi[k](x[0, j]) for k in range(p[0])])

    # insert elements of subsequent cores
    for i in range(1, d):
        for j in range(m):
            cores[i][j, :, 0, j] = np.array([phi[k](x[i, j]) for k in range(p[i])])

    # append core containing unit vectors
    cores.append(np.eye(m).reshape(m, m, 1, 1))

    # construct tensor train
    psi = scikit.TT(cores)
    return psi