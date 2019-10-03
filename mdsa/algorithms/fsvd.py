from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa as skbio_pcoa

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

        results = skbio_pcoa(DistanceMatrix(distance_matrix),
                             method='fsvd',
                             number_of_dimensions=num_dimensions_out)
        return results.samples.values, results.eigvals
