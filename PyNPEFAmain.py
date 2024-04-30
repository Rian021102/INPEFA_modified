from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import norm
from cvxopt import matrix, spmatrix, solvers
import numpy as np
from scipy import signal
from cvxopt import matrix
from spectrum.burg import _arburg2
import matplotlib.pyplot as plt

def l1tf_lm(y):
    """
    Returns an upperbound of lambda. With a regularization parameter value over lambda_max, l1tf returns the best affine fit for y.

    Parameters
    ----------
    y : numpy.ndarray or pandas.Series
        1-D array of original signal containing data with 'float' type.

    Returns
    -------
    float
        Maximum value of lambda.

    Author: Gabriel Daely
        https://github.com/daeIy

    This code is rewritten in Python 3.7 (SciPy and NumPy)
    based on l1 trend filtering algorithm by
    Kwangmoo Koh, Seung-Jean Kim and Stephen Boyd.
    https://web.stanford.edu/~boyd/papers/l1_trend_filter.html

    """
    n = len(y)
    m = n-2
    # Convert array y to cvxopt.spmatrix
    y = csr_matrix((y, (np.array([*range(n)]),
                        np.array([0]*n))))
    # Create second order difference matrix
    D = diags([1,-2,1], [0,1,2], shape=(m,n))
    DDt = D * D.T
    Dy = D * y

    return norm(spsolve(DDt, Dy), np.inf)

def l1tf(y,lambdaaa):
    """
    Finds the solution of the l1 trend estimation problem
        minimize    (1/2)*v'*D*D'*v-y'*D'*v
        subject to  ||v||_inf <= lambda
    with variable v.

    Parameters
    ----------
    y : numpy.ndarray
        1-D array of original signal containing data with 'float' type.
    lambdaaa : float
        Positive regularization parameter.

    Returns
    -------
    cvxopt.base.matrix
        Primal optimal point.

    Author: Gabriel Daely
        https://github.com/daeIy

    This code is rewritten in Python 3.7 (CVXOPT and NumPy)
    based on l1 trend filtering algorithm by
    Kwangmoo Koh, Seung-Jean Kim and Stephen Boyd.
    https://web.stanford.edu/~boyd/papers/l1_trend_filter.html

    """

    n = len(y)
    m = n-2

    # Convert array y to cvxopt.spmatrix
    y = spmatrix(y,range(n),[0]*n,tc='d')
    # Create second order difference matrix
    D = spmatrix([1,-2,1]*m,
                 [j for i in range(m) for j in [i]*3],
                 [j for i in range(m) for j in [i,i+1,i+2]],tc='d')

    # Create P and q
    P = D * D.T
    q = D * y * (-1)
    q = matrix(q)
    # Create G and h
    G = spmatrix([1]*m+[-1]*m, range(2*m), 2*[*range(m)])
    h = matrix(lambdaaa, (2*m, 1))
    # Solve the QP problem
    res = solvers.qp(P, q, G, h)
    sol = y - D.T * res['x']
    return sol


def inpefa(y,x):
    """
    Generating integrated predicition error filter analysis curve.

    Parameters
    ----------
    y : numpy.ndarray or pandas.Series
        1-D array of original curve data.
    x : numpy.ndarray
        1-D array of date/depth data.

    Returns
    -------
    ipfy : dict
        Containing original data (ipfy['OG']), long term INPEFA (ipfy['1']),
        mid term INPEFA (ipfy['2']), short term INPEFA (ipfy['3']), and
        shorter term INPEFA (ipfy['4']),

    """
    y = y.to_numpy()

    # Set maximum regularization parameter
    lambdamax = l1tf_lm(y)
    z = {}
    z['0'] = matrix(y)

    # l1 trend filtering
    for i in range(1,10):
        z['{0}'.format(i)] = l1tf(z['{0}'.format(i-1)],10**(-10+i)*lambdamax)

    # Set trend filter for long, mid, short, and shorter term
    fy = {}
    fy['1'] = z['0']-(z['1']+z['2']+z['3']+z['4']+z['5']+z['6']+z['7']+z['8'])/8.0
    fy['2'] = z['0']-(z['1']+z['2']+z['3']+z['4']+z['5']+z['6'])/6.0
    fy['3'] = z['0']-(z['1']+z['2']+z['3']+z['4']+z['5'])/5.0
    fy['4'] = z['0']-(z['1']+z['2']+z['3']+z['4'])/4.0

    # Compute Burg Filter, Prediction Error, and Integrated Prediction Error
    ipfy = {}
    ipfy['OG'] = y
    for j in range(1,5):
        # Burg's AR coeff
        bffy = _arburg2(fy['{0}'.format(j)],32)[0].real
        # PEFA
        pffy = signal.convolve(fy['{0}'.format(j)],
                               np.reshape(bffy,(len(bffy), 1)),
                               mode='same')
        # INPEFA
        iipfy = np.cumsum(pffy[::-1])[::-1]
        # Normalized to -1 <= INPEFA <= 1
        ipfy['{0}'.format(j)] = iipfy / max(abs(iipfy))

    # Plot the INPEFA curves
    plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

    plt.subplot(151)
    plt.plot(ipfy['OG'],-x) # Original signal

    plt.subplot(152)
    plt.plot(ipfy['1'],-x) # Long term INPEFA

    plt.subplot(153)
    plt.plot(ipfy['2'],-x) # Mid term INPEFA

    plt.subplot(154)
    plt.plot(ipfy['3'],-x) # Short term INPEFA

    plt.subplot(155)
    plt.plot(ipfy['4'],-x) # Shorter term INPEFA

    plt.show()

    return ipfy