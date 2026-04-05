import numpy as np
from scipy.sparse import csr_matrix

def compute_ppmi(cooccurrence: csr_matrix) -> csr_matrix:
    """
    Compute PPMI from a cooc matrix.
    """
    cooc = cooccurrence.copy().astype(np.float64)
    total = cooc.sum()

    row_sum = np.array(cooc.sum(axis = 1)).flatten()
    col_sum = np.array(cooc.sum(axis = 0)).flatten()

    cooc_coo = cooc.tocoo()
    rows, cols, data = cooc_coo.row, cooc_coo.col, cooc_coo.data
    pmi = np.log2(
        (data * total) / (row_sum[rows] * col_sum[cols])
    )

    pmi = np.maximum(pmi, 0)
    mask = pmi > 0
    ppmi_matrix = csr_matrix(
        (pmi[mask], (rows[mask], cols[mask])),
        shape = cooc.shape,
    )
    return ppmi_matrix
