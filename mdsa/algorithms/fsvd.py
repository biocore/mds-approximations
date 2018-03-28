from numpy import dot, hstack
from numpy.linalg import qr, svd
from numpy.random import standard_normal

from mdsa.algorithm import Algorithm


class FSVD(Algorithm):
    def __init__(self):
        super(FSVD, self).__init__(algorithm_name='fsvd')

    def run(self, distance_matrix, num_dimensions_out=10,
            use_power_method=False, num_levels=1):
        """
        Performs a singular value decomposition.

        Parameters
        ----------
        distance_matrix: np.array
            Numpy matrix representing the distance matrix for which the
            eigenvectors and eigenvalues shall be computed
        num_dimensions_out: int
            Number of dimensions to keep. Must be lower than or equal to the
            rank of the given distance_matrix.
        num_levels: int
            Number of levels of the Krylov method to use (see paper).
            For most applications, num_levels=1 or num_levels=2 is sufficient.
        use_power_method: bool
            Changes the power of the spectral norm, thus minimizing
            the error). See paper p11/eq8.1 DOI = {10.1137/100804139}

        Returns
        -------
        np.array
            Array of eigenvectors, each with num_dimensions_out length.
        np.array
            Array of eigenvalues, a total number of num_dimensions_out.

        Notes
        -----
        The algorithm is based on 'An Algorithm for the Principal
        Component analysis of Large Data Sets'
        by N. Halko, P.G. Martinsson, Y. Shkolnisky, and M. Tygert.
        Original Paper: https://arxiv.org/abs/1007.5510

        Ported from reference MATLAB implementation: https://goo.gl/JkcxQ2
        """
        super(FSVD, self).run(distance_matrix, num_dimensions_out)

        m, n = distance_matrix.shape

        # Note: this transpose is removed for performance, since we
        # only expect square matrices.
        # Take (conjugate) transpose if necessary, because it makes H smaller,
        # leading to faster computations
        # if m < n:
        #     distance_matrix = distance_matrix.transpose()
        #     m, n = distance_matrix.shape
        if m != n:
            raise ValueError('FSVD.run(...) expects square distance matrix')

        k = num_dimensions_out + 2

        # Form a real nxl matrix G whose entries are independent,
        # identically distributed
        # Gaussian random variables of
        # zero mean and unit variance
        G = standard_normal(size=(n, k))

        if use_power_method:
            # use only the given exponent
            H = dot(distance_matrix, G)

            for x in range(2, num_levels + 2):
                # enhance decay of singular values
                # note: distance_matrix is no longer transposed, saves work
                # since we're expecting symmetric, square matrices anyway
                # (Daniel McDonald's changes)
                H = dot(distance_matrix, dot(distance_matrix, H))

        else:
            # compute the m x l matrices H^{(0)}, ..., H^{(i)}
            # Note that this is done implicitly in each iteration below.
            H = dot(distance_matrix, G)
            # Again, removed transpose: dot(distance_matrix.transpose(), H)
            # to enhance performance
            H = hstack(
                (H, dot(distance_matrix, dot(distance_matrix, H))))
            for x in range(3, num_levels + 2):
                # Removed this transpose: dot(distance_matrix.transpose(), H)
                tmp = dot(distance_matrix, dot(distance_matrix, H))

                # Removed this transpose: dot(distance_matrix.transpose(), tmp)
                H = hstack(
                    (H, dot(distance_matrix, dot(distance_matrix, tmp))))

        # Using the pivoted QR-decomposiion, form a real m * ((i+1)l) matrix Q
        # whose columns are orthonormal, s.t. there exists a real
        # ((i+1)l) * ((i+1)l) matrix R for which H = QR
        Q, R = qr(H)

        # Compute the n * ((i+1)l) product matrix T = A^T Q
        # Removed transpose of distance_matrix for performance
        T = dot(distance_matrix, Q)  # step 3

        # Form an SVD of T
        Vt, St, W = svd(T, full_matrices=False)
        W = W.transpose()

        # Compute the m * ((i+1)l) product matrix
        Ut = dot(Q, W)

        if m < n:
            # V_fsvd = Ut[:, :num_dimensions_out] # unused
            U_fsvd = Vt[:, :num_dimensions_out]
        else:
            # V_fsvd = Vt[:, :num_dimensions_out] # unused
            U_fsvd = Ut[:, :num_dimensions_out]

        S = St[:num_dimensions_out] ** 2

        # drop imaginary component, if we got one
        eigenvalues = S.real
        eigenvectors = U_fsvd.real

        return eigenvectors, eigenvalues
