from warnings import warn

import numpy as np
import pandas as pd
from skbio import OrdinationResults
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination._utils import e_matrix, f_matrix


# - In cogent, after computing eigenvalues/vectors, the imaginary part
#   is dropped, if any. We know for a fact that the eigenvalues are
#   real, so that's not necessary, but eigenvectors can in principle
#   be complex (see for example
#   http://math.stackexchange.com/a/47807/109129 for details) and in
#   that case dropping the imaginary part means they'd no longer be
#   so, so I'm not doing that.
from mdsa.algorithm import Algorithm


def pcoa(distance_matrix, algorithm, num_dimensions_out=10):
    """Perform Principal Coordinate Analysis using a given algorithm to do so.

    Adapted from scikit-bio.

    Principal Coordinate Analysis (PCoA) is a method similar to PCA
    that works from distance matrices, and so it can be used with
    ecologically meaningful distances like UniFrac for bacteria.
    In ecology, the euclidean distance preserved by Principal
    Component Analysis (PCA) is often not a good choice because it
    deals poorly with double zeros (Species have unimodal
    distributions along environmental gradients, so if a species is
    absent from two sites at the same site, it can't be known if an
    environmental variable is too high in one of them and too low in
    the other, or too low in both, etc. On the other hand, if an
    species is present in two sites, that means that the sites are
    similar.).
    Parameters
    ----------
    algorithm : Algorithm
        Algorithm to use to decompose matrix into eigenvectors and eigenvalues
    num_dimensions_out
        k number of dimensions to return: selects k eigenvectors
        corresponding to the k largest eigenvalues
    distance_matrix : DistanceMatrix
        A distance matrix.
    Returns
    -------
    OrdinationResults
        Object that stores the PCoA results, including eigenvalues, the
        proportion explained by each of them, and transformed sample
        coordinates.
    See Also
    --------
    OrdinationResults
    Notes
    -----
    It is sometimes known as metric multidimensional scaling or
    classical scaling.
    .. note::
       If the distance is not euclidean (for example if it is a
       semimetric and the triangle inequality doesn't hold),
       negative eigenvalues can appear. There are different ways
       to deal with that problem (see Legendre & Legendre 1998, \S
       9.2.3), but none are currently implemented here.
       However, a warning is raised whenever negative eigenvalues
       appear, allowing the user to decide if they can be safely
       ignored.
    """
    if algorithm is None or not isinstance(algorithm, Algorithm):
        raise ValueError('Must specify algorithm and ensure it is a subclass'
                         ' of Algorithm.')

    # If distance_matrix is a raw numpy array representing a matrix, then
    # coerce it to scikitbio DistanceMatrix object
    if not isinstance(distance_matrix, DistanceMatrix):
        distance_matrix = DistanceMatrix(distance_matrix)

    # Implemented as per algorithm outlined in
    # Numerical Ecology (Legendre & Legendre 1998)
    # See Chapter 9, Equation 9.20
    E_matrix = e_matrix(distance_matrix.data)

    # FYI: If the used distance was euclidean, pairwise distances
    # needn't be computed from the data table Y because F_matrix =
    # Y.dot(Y.T) (if Y has been centred).
    # But since we're expecting distance_matrix to be non-euclidian,
    # we do the following computation as per
    # Numerical Ecology (Legendre & Legendre 1998)
    # See Chapter 9, Equation 9.21
    # ... which centers the matrix (a requirement for PcoA)
    F_matrix = f_matrix(E_matrix)

    # Run the given algorithm that decomposes the matrix into eigenvectors
    # and eigenvalues.
    eigenvectors, eigenvalues = algorithm.run(F_matrix, num_dimensions_out)

    # Coerce to numpy array just in case
    eigenvectors = np.array(eigenvectors)
    eigenvalues = np.array(eigenvalues)

    # Ensure eigenvectors are normalized
    eigenvectors = np.apply_along_axis(lambda vec: vec / np.linalg.norm(vec),
                                       axis=1, arr=eigenvectors)

    # Generate axis labels for output
    axis_labels = ['PC%d' % i for i in range(1, len(eigenvectors) + 1)]
    axis_labels = axis_labels[:num_dimensions_out]

    # Some algorithms do not return eigenvalues. Thus, we cannot compute
    # the array of proportion of variance explained and we cannot sort the
    # eigenvectors by their corresponding eigenvalues.
    if np.all(np.isnan(eigenvalues)):
        # Only return an OrdinationResults object wrapping the result's
        # eigenvectors. Leave the eigenvalues as NaNs.

        # Since there are no eigenvalues, we cannot compute proportions
        # explained. However, we cannot simply omit the proportions explained
        # array, since other scripts may be expecting it
        # when reading an OrdinationResults object.
        # So instead, return array of NaNs.
        proportion_explained = np.full(num_dimensions_out, np.nan)

        return OrdinationResults(
            short_method_name='PCoA',
            long_method_name='Principal Coordinate Analysis',
            samples=pd.DataFrame(eigenvectors, index=distance_matrix.ids,
                                 columns=axis_labels),
            eigvals=pd.Series(eigenvalues),
            proportion_explained=pd.Series(proportion_explained,
                                           index=axis_labels))
    else:
        # cogent makes eigenvalues positive by taking the
        # abs value, but that doesn't seem to be an approach accepted
        # by Legendre & Legendre to deal with negative eigenvalues.
        # We raise a warning in that case.

        # First, we coerce values close to 0 to equal 0.
        indices_close_to_zero = np.isclose(eigenvalues,
                                           np.zeros(eigenvalues.shape))
        eigenvalues[indices_close_to_zero] = 0

        if np.any(eigenvalues < 0):
            warn(
                "The result contains negative eigenvalues."
                " Please compare their magnitude with the magnitude of some"
                " of the largest positive eigenvalues. If the negative ones"
                " are smaller, it's probably safe to ignore them, but if they"
                " are large in magnitude, the results won't be useful. See the"
                " Notes section for more details. The smallest eigenvalue is"
                " {0} and the largest is {1}.".format(eigenvalues.min(),
                                                      eigenvalues.max()),
                RuntimeWarning
            )

        # eigvals might not be ordered, so we order them (at least one
        # is zero).
        indices_descending = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[indices_descending]
        # Sort eigenvectors in correspondance with eigenvalues' order
        eigenvectors = eigenvectors[:, indices_descending]

        # Note that at
        # least one eigenvalue is zero because only n-1 axes are
        # needed to represent n points in an euclidean space.

        # If we return only the coordinates that make sense (i.e., that have a
        # corresponding positive eigenvalue), then Jackknifed Beta Diversity
        # won't work as it expects all the OrdinationResults to have the same
        # number of coordinates. In order to solve this issue, we return the
        # coordinates that have a negative eigenvalue as 0
        num_positive = (eigenvalues >= 0).sum()
        eigenvectors[:, num_positive:] = np.zeros(eigenvectors[:,
                                                  num_positive:].shape)
        eigenvalues[num_positive:] = np.zeros(eigenvalues[num_positive:].shape)

        # Scale eigenvalues to have length = sqrt(eigenvalue). This
        # works because our eigenvectors are normalized before doing this
        # operation.
        eigenvectors = eigenvectors * np.sqrt(eigenvalues)

        # Now remove the dimensions with the least information
        # Only select k (num_dimensions_out) first eigenvectors
        # and their corresponding eigenvalues from the sorted array
        # of eigenvectors / eigenvalues

        if len(eigenvalues) > num_dimensions_out:
            eigenvectors = eigenvectors[:, :num_dimensions_out]
            eigenvalues = eigenvalues[:num_dimensions_out]

        # Calculate the array of proportion of variance explained
        proportion_explained = eigenvalues / eigenvalues.sum()

        return OrdinationResults(
            short_method_name='PCoA',
            long_method_name='Principal Coordinate Analysis',
            eigvals=pd.Series(eigenvalues, index=axis_labels),
            samples=pd.DataFrame(eigenvectors,
                                 index=distance_matrix.ids,
                                 columns=axis_labels),
            proportion_explained=pd.Series(proportion_explained,
                                           index=axis_labels))
