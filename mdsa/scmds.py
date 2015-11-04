from mdsa.approximate_mds import principal_coordinates_analysis

def scmds(dm, k=10):
    """ Performs nystrom approximation of pca

    Parameters
    ----------
    dm : np.array
        Distance matrix
    k : int
        Number of dimensions (i.e. eigenvectors)
        to return

    Returns
    -------
    coords : np.array
        List of eigenvectors
    eigvals : np.array
        Eigenvalues
    pcnts : np.array
        Percentage of variation explained
    """
    return principal_coordinates_analysis(dm, 'scmds', k)
