from abc import abstractmethod


class Algorithm(object):
    algorithms = {}

    def __init__(self, algorithm_name):
        self.name = algorithm_name

        Algorithm.algorithms[self.name] = self

    @abstractmethod
    def run(self, distance_matrix, num_dimensions_out=10):
        """ Performs approximation of Principal Component Analysis using given algorithm

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
            Eigenvectors, i.e. an array of coordinates
        eigenvalues : np.array
            Eigenvalues
        percentages: np.array
            Percentage of variation explained per coordinate
        """
        pass

    @staticmethod
    def get_algorithm(algorithm_name):
        return Algorithm.algorithms[algorithm_name]
