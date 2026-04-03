import numpy as np
from scipy.sparse import csr_matrix


def compute_ppmi(cooccurrence: csr_matrix) -> csr_matrix:
    """Compute Positive Pointwise Mutual Information from a co-occurrence matrix.

    PPMI(i,j) = max(log2(P(i,j) / (P(i) * P(j))), 0)

    where:
        P(i,j) = M[i,j] / total
        P(i) = row_sum[i] / total
        P(j) = col_sum[j] / total
    """
    cooc = cooccurrence.copy().astype(np.float64)
    total = cooc.sum()

    row_sum = np.array(cooc.sum(axis=1)).flatten()
    col_sum = np.array(cooc.sum(axis=0)).flatten()

    # Operate only on nonzero entries
    cooc_coo = cooc.tocoo()
    rows, cols, data = cooc_coo.row, cooc_coo.col, cooc_coo.data

    # PMI = log2( (M[i,j] * total) / (row_sum[i] * col_sum[j]) )
    pmi = np.log2(
        (data * total) / (row_sum[rows] * col_sum[cols])
    )

    # Positive PMI: clamp to zero
    pmi = np.maximum(pmi, 0)

    # Remove zeros to keep sparsity
    mask = pmi > 0
    ppmi_matrix = csr_matrix(
        (pmi[mask], (rows[mask], cols[mask])),
        shape=cooc.shape,
    )

    return ppmi_matrix
