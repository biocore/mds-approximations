# mds-approximations
Repository to hold new mds approximations code. The goal is to test all the ones we have access to and just move the best to skbio.

## Data
### `./mdsa/tests/data/full_sym_matrix.txt`

A symmetrical matrix for testing: The following is a distance
matrix of 100 points making up a 16-dimensional spiral. Idea was
copied from Tzeng et al. 2008 (PMID 18394154).

Note: the objects are ordered, i.e. permuting the distances will
give better MDS approximations.