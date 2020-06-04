# G.-B. Huang, Q.-Y. Zhu, and C.-K. Siew,
# “Extreme learning machine: Theory and applications,”
# Neurocomputing, vol. 70, nos. 1–3, pp. 489–501, 2006.
import numpy as np


def m_p_inverse(A):
    """
    get Moore–Penrose Generalized Inverses by using
    SVD(Singular Value Decomposition) method

    :param A:   input 2D-matrix
    :return:    A_h(2D-matrix): M-P inverse,
                pos(str): 'left' for left inverse or 'right' for right inverse
                            or 'unknown' for unknown inverse
    """
    if len(A.shape) != 2:
        print('error in ', m_p_inverse.__name__,
              ', input length must be equal to 2.')
        return None
    m = np.min(A.shape)
    u, s, vh = np.linalg.svd(A)
    smat = np.zeros(shape=A.shape, dtype=np.complex)
    smat[:m, :m] = np.diag(s)  # SVD result, A = u @ smat @ vh

    smat_h = np.zeros(shape=A.shape[::-1], dtype=np.complex)
    smat_h[:m, :m] = np.diag(1.0 / s)
    A_h = vh.T @ smat_h @ u.T  # Moore–Penrose Generalized Inverses
    rank = np.linalg.matrix_rank(A)
    if rank == A.shape[0]:
        pos = 'right'
    elif rank == A.shape[1]:
        pos = 'left'
    else:
        pos = 'unknown'
    return A_h, pos


def sigmoid(x):
    """
    a simple sigmoidal function

    :param x:   input vector
    :return:    output vector
    """
    return np.ones(shape=x.shape) / (1 + np.exp(-x))


def normalize(x, l=0., r=1.):
    """
    normalize x into range [l, r]

    :param x:   input 1-D or 2-D array
    :param l:   left value
    :param r:   right value
    :return:    normalized 1-D or 2-D array
    """
    if len(x.shape) > 2:
        print('error in', normalize.__name__,
              ', input must be 1-D or 2-D arrays')
        return None

    if l > r:
        print('error in', normalize.__name__,
              ', left value must be smaller than right value.')
        return None

    min = np.repeat(np.expand_dims(np.min(x, 0), 0), x.shape[0], 0)
    max = np.repeat(np.expand_dims(np.max(x, 0), 0), x.shape[0], 0)
    y = (x - min) / (max - min) * (r - l) + l

    return y


print('start.')

# global parameter
L = 1024        # number of Hidden layer's neutron
m = 1           # output size

# load data
ds = np.concatenate([np.loadtxt('all_data.txt')], axis=0)
permu = np.random.permutation(ds.shape[0])  # shuffle
n = ds.shape[1] - m                         # input size
N = np.int(len(permu) * 0.8)                # training num
X = ds[permu[:N], :n]                       # training sample
T = ds[permu[:N], -1]                       # training label
N_test = ds.shape[0] - N                    # test num
X_test = ds[permu[N:], :n]                  # test sample
T_test = ds[permu[N:], -1]                  # test label

# normalize the inputs(attributes) and outputs(target) into range [0, 1] and [-1, 1] respectively
X = normalize(X)
T = normalize(T, -1, 1)
X_test = normalize(X_test)
T_test = normalize(T_test, -1, 1)

# initialize the weight and bias randomly
W = np.random.rand(n, L)                    # weight between input layer and hidden layer
b = np.random.rand(1, L)                    # bias of hidden neutrons
H = sigmoid(X @ W + np.repeat(b, N, 0))     # get the output of hidden layer
H_h, pos = m_p_inverse(H)                   # pos == 'left' when L < N
beta = H_h @ T                              # compute beta(weight between hidden layer and output layer)

# predict
T_pred = sigmoid(X_test @ W + np.repeat(b, N_test, 0)) @ beta
L1 = np.linalg.norm(T_test - T_pred, ord=1) / N_test
L2 = np.linalg.norm(T_test - T_pred, ord=2) / N_test
print('L1 error:', L1)
print('L2 error:', L2)

print('done.')
