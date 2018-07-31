import torch as t
import numpy as np
import logging

device = t.device('cuda')

def Hbeta(D, beta=1.0):
    """ pytorch implementation

        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    :type D: Tensor

    """
    P = t.exp(-D * beta)
    sumP = t.sum(P)
    H = t.log(sumP) + beta * t.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X, tol=1e-5, perplexity=30.0):
    """ pytorch impl

    :type X: torch.Tensor

    Performs a binary search to get P-values in such a way that each
    conditional Gaussian has the same perplexity.
    """
    logging.debug("Computing pairwise distances...")

    # Initialize some variables
    (n, d) = X.shape
    sum_X = t.sum(X * X, 1)
    _tmp1 = t.mm(X, t.transpose(X, 0, 1))
    _tmp2 = t.add(-2 * _tmp1, sum_X)
    D = t.add(t.transpose(_tmp2, 0, 1), sum_X)
    P = t.zeros((n, n)).to(device)
    beta = t.ones((n, 1)).to(device)
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # debug print
        if i % 500 == 0:
            logging.debug("Computing P-values for point %d of %d..." % (i, n))

        # compute the Gaussian kernel and entropy for the current precision
        betamin = -float('inf')
        betamax = float('inf')
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0

        while t.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax == float('inf') or betamax == -float('inf'):
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin == float('inf') or betamin == -float('inf'):
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # return final P-matrix
    logging.debug("Mean value of sigma: %f" % t.mean(t.sqrt(1 / beta)))
    return P


def pca(X, no_dims=50):
    """ pytorch impl
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    :type X: torch.Tensor
    """
    logging.debug("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - t.mean(X, 0).repeat((n, 1))
    (l, M) = t.eig(t.mm(X.transpose(0, 1), X), eigenvectors=True)
    Y = t.mm(X, M[:, 0:no_dims])
    return Y


def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0,
         max_iter=1000, initial_momentum=0.5,
         final_momentum=0.8, eta=500,
         min_gain=0.01):
    """ pytorch impl
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.

    :type X: torch.Tensor
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    X = pca(X, initial_dims) # not sure whether pytorch has complex number or not
    (n, d) = X.shape
    Y = t.randn(n, no_dims).to(device)
    dY = t.zeros((n, no_dims)).to(device)
    iY = t.zeros((n, no_dims)).to(device)
    gains = t.ones((n, no_dims)).to(device)

    # compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + t.transpose(P, 0, 1)
    P = P / t.sum(P)
    P = P * 4.
    P = t.clamp(P, min=1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = t.sum(Y * Y, 1)
        num = -2. * t.mm(Y, Y.transpose(0, 1))
        num = 1. / (1. + t.add(t.add(num, sum_Y).transpose(0, 1), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / t.sum(num)
        Q = t.clamp(Q, min=1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            expanded_PQ = (PQ[:, i] * num[:, i]).repeat(
                (no_dims, 1)).transpose(0, 1)
            dY[i, :] = t.sum(expanded_PQ * (Y[i, :] - Y), 0)

        # perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).float() + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.)).float()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - (t.mean(Y, 0)).repeat((n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = t.sum(P * t.log(P / Q))
            logging.debug("Iteration %d: error is %f" % (iter + 1, C))

        # stop lying about P-values
        if iter == 100:
            P = P / 4.

    # return
    return Y


if __name__ == '__main__':
    """"""
    logging.basicConfig(level=logging.DEBUG)
    print('test')
    X = np.loadtxt("mnist2500_X.txt")

    X = t.tensor(X, dtype=t.float32).to(device)
    labels = np.loadtxt("mnist2500_labels.txt")
    import time
    start_time = time.time()
    Y = tsne(X, 2, 50, 20.0)
    print("--- %s seconds ---" % (time.time() - start_time))

    import pylab
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.show()
