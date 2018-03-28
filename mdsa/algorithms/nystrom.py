import random
import time
from math import log

import numpy as np
from numpy import argsort
from numpy import sqrt, power, array
from numpy import zeros
from numpy.linalg import eigh

from mdsa.algorithm import Algorithm

"""
===================== Nystrom =====================

Approximates an MDS mapping / a (partial) PCoA solution of an
(unknown) full distance matrix using a k x n seed distance matrix.

Use if you have a very high number of objects or if each distance
caculation is expensive. Speedup comes from two factors: 1. Not all
distances are calculated but only k x n. 2. Eigendecomposition is only
applied to a k x k matrix.

Calculations done after Platt (2005). See
http://research.microsoft.com/apps/pubs/?id=69185 :
``This paper unifies the mathematical foundation of three
multidimensional scaling algorithms: FastMap, MetricMap, and Landmark
MDS (LMDS). All three algorithms are based on the Nystrom
approximation of the eigenvectors and eigenvalues of a matrix. LMDS is
applies the basic Nystrom approximation, while FastMap and MetricMap
use generalizations of Nystrom, including deflation and using more
points to establish an embedding. Empirical experiments on the Reuters
and Corel Image Features data sets show that the basic Nystrom
approximation outperforms these generalizations: LMDS is more accurate
than FastMap and MetricMap with roughly the same computation and can
become even more accurate if allowed to be slower.``


Assume a full distance matrix D for N objects:

        /  E  |  F \
    D = |-----|----|
        \ F.t |  G /


The correspondong association matrix or centered inner-product
matrix K is:

        /  A  |  B \
    K = |-----|----|
        \ B.t |  C /

where A and B are computed as follows

    A_ij = - 0.50 * (E_ij^2 -
                 1/m SUM_p E_pj^2 -
                 1/m SUM_q E_iq^2 +
                 1/m^2 SUM_q E_pq^2

B is computed as in Landmark MDS, because it's simpler and works
better according to Platt):

    B_ij = - 0.50 * (F_ij^2 - 1/m SUM_q E_iq^2)


In order to approximate an MDS mapping for full matrix you only need E
and F from D as seed matrix. This will mimick the distances for m seed
objects. E is of dimension m x m and F of m x (N-m)

E and F are then used to approximate and MDS solution x for the full
distance matrix:


x_ij =  sqrt(g_j) * U_ij, if i<=m
        and
        SUM_p B_pi U_pj / sqrt(g_j)

where U_ij is the i'th component of the jth eigenvector of A (see
below) and g_j is the j'th eigenvalue of A.  The index j only runs
from 1 to k in order to make a k dimensional embedding.
"""


class Nystrom(Algorithm):
    def __init__(self):
        super(Nystrom, self).__init__(algorithm_name='nystrom')

    def run(self, distance_matrix, num_dimensions_out=10):
        super(Nystrom, self).run(distance_matrix, num_dimensions_out)

        n_seeds = int(log(distance_matrix.shape[0], 2)) ** 2
        if n_seeds <= num_dimensions_out:
            n_seeds = num_dimensions_out + 1
        elif n_seeds >= distance_matrix.shape[0]:
            n_seeds = distance_matrix.shape[0] - 1

        # Randomly sample n_seeds entries from axis 0
        sample_distmtx = distance_matrix[
            np.random.randint(0, distance_matrix.shape[0], n_seeds)]

        eigenvectors = self._nystrom(sample_distmtx, num_dimensions_out)

        eigenvalues = np.full(num_dimensions_out, np.nan)
        return array(eigenvectors), array(eigenvalues)

    def _nystrom(self, seed_distmat, dimensions, PRINT_TIMINGS=False):
        """Computes an approximate MDS mapping of an (unknown) full distance
        matrix using a kxn seed distance matrix.

        Returned matrix has the shape seed_distmat.shape[1] x dim
        """

        if not seed_distmat.shape[0] < seed_distmat.shape[1]:
            raise ValueError("seed distance matrix should "
                             "have less rows than column")
        if not dimensions <= seed_distmat.shape[0]:
            raise ValueError("number of rows of seed matrix must be "
                             ">= requested dim")

        nseeds = seed_distmat.shape[0]
        nfull = seed_distmat.shape[1]

        # matrix E: extract columns 1--nseed
        matrix_e = seed_distmat[:, 0:nseeds]

        # matrix F: extract columns nseed+1--end
        matrix_f = seed_distmat[:, nseeds:]

        # matrix A
        t0 = time.clock()
        matrix_a = self._calc_matrix_a(matrix_e)
        if PRINT_TIMINGS:
            print(("TIMING(%s): Computation of A took %f CPU secs" %
                  (__name__, time.clock() - t0)))

        # matrix B
        t0 = time.clock()
        matrix_b = self._calc_matrix_b(matrix_e, matrix_f)
        if PRINT_TIMINGS:
            print(("TIMING(%s): Computation of B took %f CPU secs" %
                  (__name__, time.clock() - t0)))

        # print("INFO: Eigendecomposing A")
        t0 = time.clock()
        # eigh: eigen decomposition for symmetric matrices
        # returns: w, v
        # w : ndarray, shape (M,)
        #     The eigenvalues, not necessarily ordered.
        # v : ndarray, or matrix object if a is, shape (M, M)
        #     The column v[:, i] is the normalized eigenvector corresponding
        #     to the eigenvalue w[i].
        # alternative is svd: [U, S, V] = numpy.linalg.svd(matrix_a)
        (eigval_a, eigvec_a) = eigh(matrix_a)
        if PRINT_TIMINGS:
            print(("TIMING(%s): Eigendecomposition of A took %f CPU secs" %
                  (__name__, time.clock() - t0)))

        # note: for optimization, if nystrom wins out, this is an excellent
        # cython or numba target
        # Sort descending
        ind = argsort(eigval_a)
        ind = ind[::-1]
        eigval_a = eigval_a[ind]
        eigvec_a = eigvec_a[:, ind]

        t0 = time.clock()
        result = zeros((nfull, dimensions))  # X in Platt 2005
        # Preventing negative eigenvalues by using abs value. Other option
        # is to set negative values to zero. Fabian recommends using
        # absolute values (as in SVD)
        sqrt_eigval_a = sqrt(abs(eigval_a))
        for i in range(nfull):
            for j in range(dimensions):
                if i + 1 <= nseeds:
                    val = sqrt_eigval_a[j] * eigvec_a[i, j]
                else:
                    # - original, unoptimised code-block
                    # numerator = 0.0
                    # for p in xrange(nseeds):
                    #     numerator += matrix_b[p, i-nseeds] * eigvec_a[p, j]
                    # val = numerator / sqrt_eigval_a[j]
                    #
                    # - optimisation attempt: actually slower
                    # numerator = sum(matrix_b[p, i-nseeds] * eigvec_a[p, j]
                    #                 for p in xrange(nseeds))
                    #
                    # - slightly optimised version: twice as fast on a seedmat
                    #  of
                    #   size 750 x 3000
                    # a_mb = array([matrix_b[p, i-nseeds] for
                    # p in xrange(nseeds)])
                    # a_eva = array([eigvec_a[p, j] for p in xrange(nseeds)])
                    # val = (a_mb*a_eva).sum() / sqrt_eigval_a[j]
                    #
                    # - optmisation suggested by daniel:
                    # 100fold increase on a seedmat of size 750 x 3000
                    num = \
                        (matrix_b[:nseeds, i - nseeds] * eigvec_a[:nseeds, j])
                    val = num.sum() / sqrt_eigval_a[j]

                result[i, j] = val

        if PRINT_TIMINGS:
            print(("TIMING(%s): Actual MDS approximation took %f CPU secs" %
                  (__name__, time.clock() - t0)))

        return result

    @staticmethod
    def _calc_matrix_a(matrix_e):
        """Calculates k x k matrix a of association matrix K

        see eq (13) in Platt from symmetrical matrix E

        A_ij = - 0.50 * (E_ij^2 -
                       1/m SUM_p E_pj^2 -
                       1/m SUM_q E_iq^2 +
                       1/m^2 SUM_q E_pq^2

        Row and column centering terms (E_pj and E_iq) are identical
        because we use a k x k submatrix of a symmetrical distance

        m equals here ncols or ncols of matrix_e
        we call it nseeds

        Arguments:
        - `matrix_e`:
           k x k part of the kxn seed distance matrix
        """

        if matrix_e.shape[0] != matrix_e.shape[1]:
            raise ValueError("matrix_e should be quadratic")
        nseeds = matrix_e.shape[0]

        row_center = zeros(nseeds)
        for i in range(nseeds):
            row_center[i] = power(matrix_e[i, :], 2).sum() / nseeds

        # E should be symmetric, i.e. column and row means are identical.
        # Why is that not mentioned in the papers? To be on the safe side
        # just do this:
        # col_center = zeros(nseeds)
        # for i in range(nseeds):
        #     col_center[i] = power(matrix_e[:, i], 2).sum()/nseeds
        # or simply:
        col_center = row_center

        grand_center = power(matrix_e, 2).sum() / power(nseeds, 2)

        # E is symmetric and so is A, which is why we don't need to loop
        # over the whole thing
        #
        # FIXME: Optimize
        #
        result = zeros((nseeds, nseeds))
        for i in range(nseeds):
            for j in range(i, nseeds):
                # avoid pow(x,2). it's slow
                eij_sq = matrix_e[i, j] * matrix_e[i, j]
                result[i, j] = -0.50 * (
                    eij_sq -
                    col_center[j] -
                    row_center[i] +
                    grand_center)
                if i != j:
                    result[j, i] = result[i, j]

        return result

    @staticmethod
    def _build_seed_matrix(fullmat_dim, seedmat_dim, getdist,
                           permute_order=True, PRINT_TIMINGS=False):
        """Builds a seed matrix of shape seedmat_dim x fullmat_dim

        Returns seed-matrix and indices to restore original order (needed
        if permute_order was True)

        Arguments:
        - `fullmat_dim`:
          dimension of the unknown (square, symmetric) "input" matrix
        - `seedmat_dim`:
          requested dimension of seed matrix.
        - `getdist`:
          distance function to compute distances. should take two
          arguments i,j with an index range of 0..fullmat_dim-1
        - `permute_order`:
           if permute_order is false, seeds will be picked sequentially.
           otherwise randomly
        """

        if not seedmat_dim < fullmat_dim:
            raise ValueError("dimension of seed matrix must be smaller "
                             "than that of full matrix")
        if not callable(getdist):
            raise ValueError("distance getter function not callable")

        if permute_order:
            picked_seeds = random.sample(list(range(fullmat_dim)), seedmat_dim)
        else:
            picked_seeds = list(range(seedmat_dim))
        # assert len(picked_seeds) == seedmat_dim, (
        #    "mismatch between number of picked seeds and seedmat dim.")

        # Putting picked seeds/indices at the front is not enough,
        # need to change/correct all indices to maintain consistency
        #
        used_index_order = list(range(fullmat_dim))
        picked_seeds.sort()  # otherwise the below fails
        for i, seed_idx in enumerate(picked_seeds):
            used_index_order.pop(seed_idx - i)
        used_index_order = picked_seeds + used_index_order

        # Order is now determined in used_index_order
        # first seedmat_dim objects are seeds
        # now create seedmat
        #
        t0 = time.clock()
        seedmat = zeros((len(picked_seeds), fullmat_dim))
        for i in range(len(picked_seeds)):
            for j in range(fullmat_dim):
                if i < j:
                    seedmat[i, j] = getdist(used_index_order[i],
                                            used_index_order[j])
                elif i == j:
                    continue
                else:
                    seedmat[i, j] = seedmat[j, i]

        restore_idxs = argsort(used_index_order)
        if PRINT_TIMINGS:
            print(("TIMING(%s): Seedmat calculation took %f CPU secs" %
                  (__name__, time.clock() - t0)))

        # Return the seedmatrix and the list of indices which can be used to
        # recreate original order
        return (seedmat, restore_idxs)

    @staticmethod
    def _calc_matrix_b(matrix_e, matrix_f):
        """Calculates k x n-k matrix b of association matrix K

        See eq (14) and (15) in Platt (2005)

        This is where Nystrom and LMDS differ: LMDS version leaves the
        constant centering term out, This simplifies the computation.

        Arguments:
        - `matrix_e`:
           k x k part of the kxn seed distance matrix
        - `matrix_f`:
           k x n-k part of the kxn seed distance matrix
        """

        (nrows, ncols) = matrix_f.shape

        if matrix_e.shape[0] != matrix_e.shape[1]:
            raise ValueError("matrix_e should be quadratic")
        if matrix_f.shape[0] != matrix_e.shape[0]:
            raise ValueError(
                "matrix_e and matrix_f should have same number of rows")

        nseeds = matrix_e.shape[0]

        # row_center_e was also precomputed in calc_matrix_a but
        # computation is cheap
        #
        row_center_e = zeros(nseeds)
        for i in range(nseeds):
            row_center_e[i] = power(matrix_e[i, :], 2).sum() / nseeds

        # The following is not needed in LMDS but part of original Nystrom
        #
        # ncols_f = matrix_f.shape[1]
        # col_center_f = zeros(ncols_f)
        # for i in range(ncols_f):
        #    col_center_f[i] = power(matrix_f[:, i], 2).sum()/nseeds
        #
        # subtract col_center_f[j] below from result[i, j] and you have
        # nystrom. dont subtract it and you have lmds

        # result = zeros((nrows, ncols))
        # for i in xrange(nrows):
        #    for j in xrange(ncols):
        #
        #        # - original one line version:
        #        #
        #        #result[i, j] = -0.50 * (
        #        #    power(matrix_f[i, j], 2) -
        #        #    row_center_e[i])
        #        #
        #        # - optimised version avoiding pow(x,2)
        #        #   3xfaster on a 750x3000 seed-matrix
        #        fij = matrix_f[i, j]
        #        result[i, j] = -0.50 * (
        #            fij * fij -
        #            row_center_e[i])
        #
        # - optimised single line version of the code block above. pointed out
        #   by daniel. 20xfaster on a 750x3000 seed-matrix. cloning idea
        #   copied from
        #   stackoverflow.com/questions/1550130/cloning-row-or-column-vectors
        result = -0.5 * (matrix_f ** 2 - array([row_center_e, ] * ncols)
                         .transpose())

        return result
