import numpy as np


def center_distance_matrix(distance_matrix):
    """
    # FYI: If the used distance was euclidean, pairwise distances
    # needn't be computed from the data table Y because F_matrix =
    # Y.dot(Y.T) (if Y has been centred).
    # But since we're expecting distance_matrix to be non-euclidian,
    # we do the following computation as per
    # Numerical Ecology (Legendre & Legendre 1998)
    """

    # Daniel McDonald note:
    # e and f matrix modifications validated via procrustes against pcoa using
    # ~600 samples (studies 10105 and 10564). procusted was like m^2 of 0.000
    # using these
    # rewrites and the originals from skbio
    return f_matrix_optimized(e_matrix_optimized(distance_matrix))


def e_matrix_optimized(distance_matrix):
    """
    Compute E matrix from a distance matrix.
    Squares and divides by -2 the input elementwise. Eq. 9.20 in
    Legendre & Legendre 1998.
    """

    # modified from skbio, performing row-wise to avoid excessive memory
    # allocations
    for i in np.arange(len(distance_matrix)):
        distance_matrix[i] = (distance_matrix[i] * distance_matrix[i]) / -2
    return distance_matrix


def f_matrix_optimized(e_matrix):
    """
    Compute F matrix from E matrix.
    Centring step: for each element, the mean of the corresponding
    row and column are substracted, and the mean of the whole
    matrix is added. Eq. 9.21 in Legendre & Legendre 1998.
    """

    # modified from skbio, performing rowwise to avoid excessive memory
    # allocations
    row_means = np.zeros(len(e_matrix), dtype=float)
    col_means = np.zeros(len(e_matrix), dtype=float)
    matrix_mean = 0.0

    for i in np.arange(len(e_matrix)):
        row_means[i] = e_matrix[i].mean()
        matrix_mean += e_matrix[i].sum()
        col_means += e_matrix[i]
    matrix_mean /= len(e_matrix) ** 2
    col_means /= len(e_matrix)

    for i in np.arange(len(e_matrix)):
        v = e_matrix[i]
        v -= row_means[i]
        v -= col_means
        v += matrix_mean
        e_matrix[i] = v
    return e_matrix
