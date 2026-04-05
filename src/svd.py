import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def compute_embeddings(ppmi_matrix: csr_matrix, k: int = 300) -> dict:
    """
    Compute truncated SVD of the PPMI matrix.
    Uses scipy.sparse.linalg.svds for efficiency.
    Returns:
        dict with keys "U" (V x k), "S" (k,), "Vt" (k x V),
        sorted by singular values in descending order.
    """
    U, S, Vt = svds(ppmi_matrix, k = k)
    order = np.argsort(-S)
    U = U[:, order]
    S = S[order]
    Vt = Vt[order, :]

    return {"U": U, "S": S, "Vt": Vt}


def get_word_vectors(U: np.ndarray, S: np.ndarray, variant: str = "sqrt") -> np.ndarray:
    """
    Extract word vectors from SVD components.
    Variants:
        "plain": U_k - raw left singular vectors
        "weighted": U_k @ diag(S_k) - fully weighted
        "sqrt": U_k @ diag(√S_k - symmetric weighting (default from report)
    """
    if variant ==  "plain":
        return U
    elif variant ==  "weighted":
        return U * S[np.newaxis, :]
    elif variant ==  "sqrt":
        return U * np.sqrt(S)[np.newaxis, :]
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'plain', 'weighted', or 'sqrt'.")
