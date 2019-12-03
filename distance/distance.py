import numpy as np
from numpy.linalg import norm


def _validate_vector(u, dtype=None):
    u = np.asarray(u, dtype=dtype, order='c').squeeze()
    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u


def cosine(u, v):
    """
    Computes the Cosine distance between 1-D arrays.
    The Cosine distance between `u` and `v`, is defined as
    .. math::
       1 - \\frac{u \\cdot v}
                {||u||_2 ||v||_2}.
    where :math:`u \\cdot v` is the dot product of :math:`u` and
    :math:`v`.
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    Returns
    -------
    cosine : double
        The Cosine distance between vectors `u` and `v`.
    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    dist = np.dot(u, v) / (norm(u) * norm(v))
    return dist


def jaccard(u, v):
    """
    Computes the Jaccard-Needham dissimilarity between two boolean 1-D arrays.
    The Jaccard-Needham dissimilarity between 1-D boolean arrays `u` and `v`,
    is defined as
    .. math::
       \\frac{c_{TF} + c_{FT}}
            {c_{TT} + c_{FT} + c_{TF}}
    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`.
    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.
    Returns
    -------
    jaccard : double
        The Jaccard distance between vectors `u` and `v`.
    """
    u = _validate_vector(u)
    v = _validate_vector(v)

    intersection = np.dot(u, v)

    a = pow(norm(u), 2)
    b = pow(norm(v), 2)

    return intersection / (a + b - intersection)


def dice(u, v):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    u : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    v : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `u` and `v` are switched.
    """

    u = _validate_vector(u)
    v = _validate_vector(v)

    a = pow(norm(u), 2)
    b = pow(norm(v), 2)

    # Compute Dice coefficient
    intersection = np.dot(u, v)

    return 2 * intersection / (a + b)


def euclidean(u, v):
    """
    Computes the Euclidean distance between two 1-D arrays.
    The Euclidean distance between 1-D arrays `u` and `v`, is defined as
    .. math::
       {||u-v||}_2
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    Returns
    -------
    euclidean : double
        The Euclidean distance between vectors `u` and `v`.
    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    dist = norm(u - v)
    return dist


def manhattan(u, v):
    """
    Computes the City Block (Manhattan) distance.
    Computes the Manhattan distance between two 1-D arrays `u` and `v`,
    which is defined as
    .. math::
       \\sum_i {\\left| u_i - v_i \\right|}.
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    Returns
    -------
    cityblock : double
        The City Block (Manhattan) distance between vectors `u` and `v`.
    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    return abs(u - v).sum()
