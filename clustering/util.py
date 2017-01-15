import numpy as np
from scipy.special import digamma

def log_choose(n, k):
    """
    ln(nCk)
    """
    return np.sum(np.log(range(1, n+1))) - np.sum(np.log(range(1, k+1))) - np.sum(np.log(range(1, n-k+1)))

"""
FROM BNPY
"""
def calcBetaExpectations(eta1, eta0):
    ''' Evaluate expected value of log u under Beta(u | eta1, eta0)

    Returns
    -------
    ElogU : 1D array, size K
    Elog1mU : 1D array, size K
    '''
    digammaBoth = digamma(eta0 + eta1)
    ElogU = digamma(eta1) - digammaBoth
    Elog1mU = digamma(eta0) - digammaBoth
    return ElogU, Elog1mU


def inplaceExpAndNormalizeRows(R, minVal=1e-40):
    ''' Compute exp(R), normalize rows to sum to one, and set min val.

    Post Condition
    --------
    Each row of R sums to one.
    Minimum value of R is equal to minVal.
    '''
    inplaceExpAndNormalizeRows_numpy(R)
    if minVal is not None:
        np.maximum(R, minVal, out=R)


def inplaceExpAndNormalizeRows_numpy(R):
    ''' Compute exp(R), normalize rows to sum to one, and set min val.

    Post Condition
    --------
    Each row of R sums to one.
    Minimum value of R is equal to minVal.
    '''
    R -= np.max(R, axis=1)[:, np.newaxis]
    np.exp(R, out=R)
    R /= R.sum(axis=1)[:, np.newaxis]