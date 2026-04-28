import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def _norm(x: np.ndarray) -> float:
    """
    L2 norm of a vector. manual sqrt(sum of squares).
    """
    return float(np.sqrt((x * x).sum()))


def _column_norms(M: np.ndarray) -> np.ndarray:
    """
    L2 norm of each column of M.
    """
    return np.sqrt((M * M).sum(axis = 0))


def _qr_mgs(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    QR decomposition via Modified Gram-Schmidt.
    A is V x p (tall). Returns Q (V x p, orthonormal cols), R (p x p, upper triangular).
    Numerically more stable than classical Gram-Schmidt.
    """
    n, p = A.shape
    Q = A.copy().astype(np.float64)
    R = np.zeros((p, p), dtype = np.float64)

    for i in range(p):
        R[i, i] = _norm(Q[:, i])
        if R[i, i] < 1e-12:
            # degenerate column - replace with random and re-project
            Q[:, i] = np.random.randn(n)
            for j in range(i):
                Q[:, i] -= (Q[:, j] @ Q[:, i]) * Q[:, j]
            R[i, i] = _norm(Q[:, i])
        Q[:, i] /= R[i, i]
        for j in range(i + 1, p):
            R[i, j] = Q[:, i] @ Q[:, j]
            Q[:, j] -= R[i, j] * Q[:, i]

    return Q, R


def power_iteration_svd(
    A: csr_matrix,
    k: int,
    n_iter: int = 30,
    oversample: int = 10,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncated SVD via subspace (block power) iteration on A^T A.
    Uses only sparse @ dense matmul and our manual _qr_mgs / _norm.
    Right singular vectors are eigenvectors of A^T A with eigenvalues sigma^2.
    Args:
        A: V x V sparse matrix
        k: number of singular triples to return
        n_iter: power iteration steps
        oversample: extra columns for stability (Halko et al.)
        seed: for reproducibility
    Returns:
        (U, S, Vt) sorted by singular value descending
    """
    rng = np.random.default_rng(seed)
    V_dim = A.shape[1]
    p = k + oversample

    # 1. random initialization, then orthonormalize
    V = rng.standard_normal((V_dim, p))
    V, _ = _qr_mgs(V)

    # 2. block power iteration on A^T A, with re-orthonormalization
    for _ in range(n_iter):
        Y = A @ V              # sparse @ dense, V x p
        Z = A.T @ Y            # sparse @ dense, V x p
        V, _ = _qr_mgs(Z)

    # 3. recover singular values and left singular vectors
    AV = A @ V                 # V x p, columns are sigma_i * u_i
    S = _column_norms(AV)

    U = np.zeros_like(AV)
    nonzero = S > 1e-12
    U[:, nonzero] = AV[:, nonzero] / S[nonzero]
    Vt = V.T

    # 4. sort descending, trim oversample to top-k
    order = np.argsort(-S)[:k]
    return U[:, order], S[order], Vt[order, :]


def compute_embeddings(
    ppmi_matrix: csr_matrix,
    k: int = 300,
    method: str = "power_iteration",
    n_iter: int = 30,
    oversample: int = 10,
    seed: int | None = None,
) -> dict:
    """
    Compute truncated SVD of the PPMI matrix.
    method:
        "power_iteration" - our manual implementation (default)
        "lanczos" - scipy.sparse.linalg.svds, kept as a comparison oracle only
    Returns:
        dict with keys "U" (V x k), "S" (k,), "Vt" (k x V),
        sorted by singular values in descending order.
    """
    if method == "power_iteration":
        U, S, Vt = power_iteration_svd(
            ppmi_matrix, k = k, n_iter = n_iter,
            oversample = oversample, seed = seed,
        )
    elif method == "lanczos":
        U, S, Vt = svds(ppmi_matrix, k = k)
        order = np.argsort(-S)
        U = U[:, order]
        S = S[order]
        Vt = Vt[order, :]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'power_iteration' or 'lanczos'.")

    return {"U": U, "S": S, "Vt": Vt}


def get_word_vectors(U: np.ndarray, S: np.ndarray, variant: str = "sqrt") -> np.ndarray:
    """
    Extract word vectors from SVD components.
    Variants:
        "plain": U_k - raw left singular vectors
        "weighted": U_k @ diag(S_k) - fully weighted
        "sqrt": U_k @ diag(√S_k) - symmetric weighting (default from report)
    """
    if variant ==  "plain":
        return U
    elif variant ==  "weighted":
        return U * S[np.newaxis, :]
    elif variant ==  "sqrt":
        return U * np.sqrt(S)[np.newaxis, :]
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'plain', 'weighted', or 'sqrt'.")
