from abc import abstractmethod


class Algorithm(object):
    algorithms = {}

    def __init__(self, algorithm_name):
        self.name = algorithm_name

    @abstractmethod
    def run(self, distance_matrix, num_dimensions_out=10):
        """
        Performs approximation of Principal Coordinate Analysis (PCoA) using given algorithm.
        Returns the principal coordinate results (not necessarily sorted).

        Parameters
        ----------
        distance_matrix : np.array
            Distance matrix
        num_dimensions_out : int
            Number of dimensions (i.e. eigenvectors)
            to return, i.e. what dimensionality to approximate given distance matrix to.

        Returns
        -------
        eigenvectors : np.array
            Eigenvectors, where each row is an axis and the columns are points within the axis
        eigenvalues : np.array
            Eigenvalues corresponding to the rows, indicating the amount of the variation that
            the axis in that row accounts for
        percentages: np.array
            Percentage of variation explained per coordinate
        """
        pass

    @staticmethod
    def get_algorithm(algorithm_name):
        return Algorithm.algorithms[algorithm_name]

    @staticmethod
    def register(algorithm):
        if not isinstance(algorithm, Algorithm):
            raise ValueError('Expecting a subclass of Algorithm.')
        Algorithm.algorithms[algorithm.name] = algorithm
