"""Functions for doing fast multidimensional scaling of distances
using Nystrom/LMDS approximation ans Split and Combine MDS.

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

===================== SCMDS =====================

The is a Python/Numpy implementation of SCMDS:
Tzeng J, Lu HH, Li WH.
Multidimensional scaling for large genomic data sets.
BMC Bioinformatics. 2008 Apr 4;9:179.
PMID: 18394154

The basic idea is to avoid the computation and eigendecomposition of
the full pairwise distance matrix. Instead only compute overlapping
submatrices/tiles and their corresponding MDS separately. The
solutions are then joined using an affine mapping approach.

=================================================
"""


from numpy import sign, sqrt, power, mean, array
from math import log
from numpy import matrix, ones, dot, argsort, diag, eye
from numpy import zeros, concatenate, ndarray, kron
from numpy.linalg import eig, eigh, qr
from random import sample
import time

# print simple timings
PRINT_TIMINGS = False


def principal_coordinates_analysis(distance_matrix, algorithm, dimensions):
    """Takes a distance matrix and returns principal coordinate results

    point_matrix: each row is an axis and the columns are points within the
                  axis
    eigvals: correspond to the rows and indicate the amount of the variation
        that that the axis in that row accounts for
    NOT NECESSARILY SORTED
    """
    if algorithm == 'scmds':
        num_objects = distance_matrix.shape[0]
        tile_size = int(num_objects * 0.6)
        tile_overlap = int(tile_size * 0.09)

        if tile_overlap > tile_size:
            raise ValueError("The distance matrix is to small to be divided")
        if dimensions > tile_overlap:
            tile_overlap = dimensions

        comb_mds = CombineMds()
        tile_no = 1
        tile_start = 0
        tile_end = tile_size
        curr_tile_size = tile_end - tile_start

        while curr_tile_size > 0:
            (tile_eigvecs, tile_eigvals) = cmds_tzeng(
                distance_matrix[tile_start:tile_end, tile_start:tile_end],
                dimensions)

            comb_mds.add(tile_eigvecs, tile_overlap)

            if tile_end == num_objects:
                break
            else:
                tile_start = tile_start + tile_size - tile_overlap
                if tile_end + tile_size >= num_objects:
                    tile_end = num_objects
                else:
                    tile_end = tile_end + tile_size - tile_overlap

            tile_no += 1
            if tile_no > 10:
                break
            curr_tile_size = tile_end - tile_start

        coords = array(comb_mds.getFinalMDS())
        eigvals = ['NaN'] * dimensions
        pcnts = ['NaN'] * dimensions
    else:
        raise ValueError("Method (%s) not implemented", algorithm)

    return coords, eigvals, pcnts


def rowmeans(mat):
    """Returns a `column-vector` of row-means of the 2d input array/matrix
    """

    if not len(mat.shape) == 2:
        raise ValueError("argument is not a 2D ndarray")

    # nrows = mat.shape[0]
    #  create a column vector (hack!)
    # cv = matrix(arange(float(nrows)))
    # cv = cv.T
    #  for i in range(nrows):
    #     cv[i] = mat[i].mean()
    #
    # As pointed out by Daniel the above is much simpler done in Numpy:
    cv = matrix(mean(mat, axis=1).reshape((mat.shape[0], 1)))

    return cv


def affine_mapping(matrix_x, matrix_y):
    """Returns an affine mapping function.

    Arguments are two different MDS solutions/mappings (identical
    dimensions) on the the same objects/distances.

    Affine mapping function:
    Y = UX + kron(b,ones(1,n)), UU' = I
    X = [x_1, x_2, ... , x_n]; x_j \in R^m
    Y = [y_1, y_2, ... , y_n]; y_j \in R^m

    From Tzeng 2008:
    `The projection of xi,2 from q2 dimension to q1 dimension
    induces computational errors (too). To avoid this error, the
    sample number of the overlapping region is important. This
    sample number must be large enough so that the derived
    dimension of data is greater or equal to the real data`

    Notes:
    - This was called moving in scmdscale.m
    - We work on tranposes, i.e. coordinates for objects are in columns


    Arguments:
    - `matrix_x`: first mds solution (`reference`)
    - `matrix_y`: seconds mds solution

    Return:
    A tuple of unitary operator u and shifting operator b such that:
    y = ux + kron(b, ones(1,n))
    """

    if matrix_x.shape != matrix_y.shape:
        raise ValueError("input matrices are not of same size")

    if not matrix_x.shape[0] <= matrix_x.shape[1]:
        raise ValueError("input matrices should have more columns than rows")

    # Have to check if we have not more rows than columns, otherwise,
    # the qr function below might behave differently in the matlab
    # prototype, but not here. In addition this would mean that we
    # have more dimensions than overlapping objects which shouldn't
    # happen
    #
    # matlab code uses economic qr mode (see below) which we can't use
    # here because we need both return matrices.
    #
    # see
    # http://www.mathworks.com/access/helpdesk/help/techdoc/index.html?/access/helpdesk/help/techdoc/ref/qr.html
    # [Q,R] = qr(A,0) produces the economy-size decomposition.
    # If m > n, only the first n columns of Q and
    # the first n rows of R are computed.
    # If m<=n, this is the same as [Q,R] = qr(A)
    #
    # [Q,R] = qr(A), where A is m-by-n, produces
    # an m-by-n upper triangular matrix R and
    # an m-by-m unitary matrix Q so that A = Q*R
    #
    # That's why we check above with an assert

    ox = rowmeans(matrix_x)
    oy = rowmeans(matrix_y)

    mx = matrix_x - kron(ox, ones((1, matrix_x.shape[1])))
    my = matrix_y - kron(oy, ones((1, matrix_x.shape[1])))

    (qx, rx) = qr(mx)
    (qy, ry) = qr(my)

    # sign correction
    #
    # Daniel suggest to use something like
    # [arg]where(sign(a.diagonal()) != sign(b.diagonal())) and then
    # iterate over the results. Couldn't figure out how to do this
    # properly :(

    for i in range(qx.shape[1]):
        if sign(rx[i, i]) != sign(ry[i, i]):
            qy[:, i] *= -1

    # matrix multiply: use '*' as all arguments are of type matrix
    ret_u = qy * qx.transpose()
    ret_b = oy - ret_u * ox

    return (ret_u, ret_b)


def adjust_mds_to_ref(mds_ref, mds_add, n_overlap):
    """Transforms mds_add such that the overlap mds_ref and mds_add
    has same configuration.

    As overlap (n_overlap objects) we'll use the end of mds_ref
    and the beginning of mds_add

    Both matrices must be of same dimension (column numbers) but
    can have different number of objects (rows) because only
    overlap will be used.

    Arguments:
    - `mds_ref`: reference mds solution
    - `mds_add`: mds solution to adjust
    - `n_overlap`: overlap size between mds_ref and mds_add

    Return:
    Adjusted version of mds_add which matches configuration of mds_ref
    """

    if mds_ref.shape[1] != mds_add.shape[1]:
        raise ValueError("given mds solutions have different dimensions")
    if not (mds_ref.shape[0] >= n_overlap and mds_add.shape[0] >= n_overlap):
        raise ValueError("not enough overlap between given mds mappings")

    # Use transposes for affine_mapping!
    overlap_ref = mds_ref.transpose()[:, -n_overlap:]
    overlap_add = mds_add.transpose()[:, 0:n_overlap]
    (unitary_op, shift_op) = affine_mapping(overlap_add, overlap_ref)
    # paranoia: unitary_op is of type matrix, make sure mds_add
    # is as well so that we can use '*' for matrix multiplication
    mds_add_adj = unitary_op * matrix(mds_add.transpose()) + \
        kron(shift_op, ones((1, mds_add.shape[0])))
    mds_add_adj = mds_add_adj.transpose()

    return mds_add_adj


def recenter(joined_mds):
    """Recenter an Mds mapping that has been created by joining, i.e.
    move points so that center of gravity is zero.

    Note:
    Not sure if recenter is a proper name, because I'm not exactly
    sure what is happening here

    Matlab prototype from Tzeng et al. 2008:
         X = zero_sum(X); # subtract column means
         M = X'*X;
         [basis,L] = eig(M);
         Y = X*basis;
         return Y = Y(:,end:-1:1);

    Arguments:
    - `mds_combined`:

    Return:
    Recentered version of `mds_combined`
    """

    # or should we cast explictely?
    if not isinstance(joined_mds, matrix):
        raise ValueError("mds solution has to be of type matrix")

    # As pointed out by Daniel: the following two loop can be done in
    # one if you pass down the axis variable to means()
    #
    # colmean = []
    # for i in range(joined_mds.shape[1]):
    #    colmean.append(joined_mds[:, i].mean())
    # for i in range(joined_mds.shape[0]):
    #    joined_mds[i, :] = joined_mds[i, :] - colmean
    #
    joined_mds = joined_mds - joined_mds.mean(axis=0)

    matrix_m = dot(joined_mds.transpose(), joined_mds)
    (eigvals, eigvecs) = eig(matrix_m)

    # Note / Question: do we need sorting?
    # idxs_ascending = eigvals.argsort()
    # idxs_descending = eigvals.argsort()[::-1]# reverse!
    # eigvecs = eigvecs[idxs_ascending]
    # eigvals = eigvals[idxs_ascending]

    # joined_mds and eigvecs are of type matrix so use '*' for
    # matrix multiplication
    joined_mds = joined_mds * eigvecs

    # NOTE: the matlab protoype reverses the vector before
    # returning. We don't because I don't know why and results are
    # good

    return joined_mds


def combine_mds(mds_ref, mds_add, n_overlap):
    """Returns a combination of the two MDS mappings mds_ref and
    mds_add.

    This is done by finding an affine mapping on the
    overlap between mds_ref and mds_add and changing mds_add
    accordingly.

    As overlap we use the last n_overlap objects/rows in mds_ref and
    the first n_overlap objects/rows in mds_add.

    The overlapping part will be replaced, i.e. the returned
    combination has the following row numbers:
    mds_ref.nrows + mds_add.nrows - overlap

    The combined version will eventually need recentering.
    See recenter()

    Arguments:
    - `mds_ref`: reference mds mapping
    - `mds_add`: mds mapping to add
    """

    if mds_ref.shape[1] != mds_add.shape[1]:
        raise ValueError("given mds solutions have different dimensions")
    if not mds_ref.shape[0] >= n_overlap:
        raise ValueError("not enough items for overlap in mds_ref")
    if not mds_add.shape[0] >= n_overlap:
        raise ValueError("not enough items for overlap in mds_add")

    mds_add_adj = adjust_mds_to_ref(mds_ref, mds_add, n_overlap)

    combined_mds = concatenate((
        mds_ref[0:mds_ref.shape[0] - n_overlap, :], mds_add_adj))

    return combined_mds


def cmds_tzeng(distmat, dim=None):
    """Calculate classical multidimensional scaling on a distance matrix.

    Faster than default implementation of dim is smaller than
    distmat.shape

    Arguments:
    - `distmat`: distance matrix (non-complex, symmetric ndarray)
    - `dim`:     wanted dimensionality of MDS mapping (defaults to distmat dim)

    Implementation as in Matlab prototype of SCMDS, see
    Tzeng J et al. (2008), PMID: 18394154
    """

    if not isinstance(distmat, ndarray):
        raise ValueError("Input matrix is not a ndarray")
    (m, n) = distmat.shape
    if m != n:
        raise ValueError("Input matrix is not a square matrix")
    if not dim:
        dim = n

    # power goes wrong here if distmat is ndarray because of matrix
    # multiplication syntax difference between array and
    # matrix. (doesn't affect gower's variant). be on the safe side
    # and convert explicitely (it's local only):
    distmat = matrix(distmat)

    h = eye(n) - ones((n, n)) / n
    assocmat = -h * (power(distmat, 2)) * h / 2
    # print "DEBUG assocmat[:3] = %s" % assocmat[:3]

    (eigvals, eigvecs) = eigh(assocmat)

    # Recommended treatment of negative eigenvalues (by Fabian): use
    # absolute value (reason: SVD does the same)
    eigvals = abs(eigvals)

    ind = argsort(eigvals)[::-1]
    eigvals = eigvals[ind]
    eigvecs = eigvecs[:, ind]
    eigvals = eigvals[:dim]
    eigvecs = eigvecs[:, :dim]

    eigval_diagmat = matrix(diag(sqrt(eigvals)))
    eigvecs = eigval_diagmat * eigvecs.transpose()
    return (eigvecs.transpose(), eigvals)


class CombineMds(object):
    """
    Convinience class for joining MDS mappings. Several mappings can
    be added.

    The is uses the above Python/Numpy implementation of SCMDS.
    See Tzeng  et al. 2008, PMID: 18394154
    """

    def __init__(self, mds_ref=None):
        """
        Init with reference MDS
        """

        self._mds = mds_ref
        self._need_centering = False

    def add(self, mds_add, overlap_size):
        """Add a new MDS mapping to existing one
        """

        if self._mds is None:
            self._mds = mds_add
            return

        if not self._mds.shape[0] >= overlap_size:
            raise ValueError("not enough items for overlap in reference mds")
        if not mds_add.shape[0] >= overlap_size:
            raise ValueError("not enough items for overlap in mds to add")

        self._need_centering = True
        self._mds = combine_mds(self._mds, mds_add, overlap_size)

    def getFinalMDS(self):
        """Get final, combined MDS solution
        """

        if self._need_centering:
            self._mds = recenter(self._mds)

        return self._mds


