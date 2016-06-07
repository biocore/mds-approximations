"""Functions for doing fast multidimensional scaling of distances
using Nystrom/LMDS approximation ans Split and Combine MDS.



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


