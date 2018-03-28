from numpy import dot
from numpy.random import standard_normal
from scipy.linalg import qr
from scipy.sparse.linalg import eigsh

from mdsa.algorithm import Algorithm


class SSVD(Algorithm):
    def __init__(self):
        super(SSVD, self).__init__(algorithm_name='ssvd')

    def run(self, distance_matrix, num_dimensions_out=10, p=10, qiter=0):
        """
            Takes a distance matrix and returns eigenvalues and eigenvectors,
            using SSVD routine.

            Parameters
            ----------
            distance_matrix: np.array
                distance matrix to compute eigenvalues and eigenvectors for.
            num_dimensions_out: int
                number of dimensions to reduce to
            p: int
                oversampling parameter that is added
                to num_dimensions_out to boost accuracy
            qiter:
                number of iterations to go through to boost accuracy

            Returns
            -------
            np.array
                Array of eigenvectors, each with num_dimensions_out length.
            np.array
                Array of eigenvalues, a total number of num_dimensions_out.

            Notes
            -----
            The algorithm for SSVD is outlined in:
            'Finding Structure with Randomness: Probabilistic Algorithms
            for Constructing Approximate Matrix Decompositions'
            by N. Halko, P.G. Martinsson, and J.A. Tropp

            Code adapted from R: https://goo.gl/gSPNZh
        """
        super(SSVD, self).run(distance_matrix, num_dimensions_out)

        m, n = distance_matrix.shape

        # an mxn matrix M has at most p = min(m,n) unique
        p = min(min(m, n) - num_dimensions_out, p)

        # singular values
        r = num_dimensions_out + p  # rank plus oversampling parameter p

        omega = standard_normal(size=(n, r))  # generate random matrix omega

        # compute a sample matrix Y: apply distance_matrix to random
        # vectors to identify part of its range corresponding
        # to largest singular values
        y = dot(distance_matrix, omega)

        # find an ON matrix st. Y = QQ'Y
        Q, R = qr(y)

        # multiply distance_matrix by Q whose columns form
        # an orthonormal basis for the range of Y
        b = dot(Q.transpose(), distance_matrix)

        # often, no iteration required to small error in eqn. 1.5
        for i in range(1, qiter):
            y = dot(distance_matrix, b.transpose())
            Q, R = qr(y)
            b = dot(Q.transpose(), distance_matrix)

        # compute eigenvalues of much smaller matrix bbt
        bbt = dot(b, b.transpose())

        # NOTE: original code defaulted to eigh when eigsh was unavailable.
        # Since it claimed that eigsh is faster and we always have it available
        # through scipy, we compute using eigsh.
        eigenvalues, eigenvectors = eigsh(bbt, num_dimensions_out)

        U_ssvd = dot(Q, eigenvectors)  # [:,1:k]

        # don't need to compute V. Just in case, here's how:
        # V_ssvd = dot(transpose(b),dot(eigvecs, diag(1/eigvals,0))) [:, 1:k]

        eigenvectors = U_ssvd.real

        return eigenvectors, eigenvalues
