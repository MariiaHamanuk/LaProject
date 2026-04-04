from collections import Counter
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def load_corpus(path: str) -> list[str]:
    """Read text8 file and return list of tokens."""
    with open(path, "r") as f:
        text = f.read()
    return text.strip().split()


def build_vocab(tokens: list[str], max_vocab: int = 10_000) -> tuple[dict, list]:
    """Build vocabulary from the most frequent tokens.

    Returns:
        word2idx: dict mapping word -> index
        idx2word: list where idx2word[i] = word
    """
    counts = Counter(tokens)
    most_common = counts.most_common(max_vocab)
    idx2word = [word for word, _ in most_common]
    word2idx = {word: i for i, word in enumerate(idx2word)}
    return word2idx, idx2word


def build_cooccurrence(
    tokens: list[str], word2idx: dict, window: int = 5
) -> csr_matrix:
    """Build sparse symmetric co-occurrence matrix.

    For each token in the corpus, count how many times each other token
    appears within a symmetric window of size `window` on each side.
    Only tokens present in word2idx are counted.
    """
    V = len(word2idx)
    N = len(tokens)

    print("Mapping tokens to indices...")
    token_ids = np.array(
        [word2idx.get(t, -1) for t in tokens], dtype=np.int32
    )

    all_rows = []
    all_cols = []

    print(f"Collecting co-occurrence pairs (window={window})...")
    for offset in range(1, window + 1):
        # offset to the right:  token[i] <-> token[i + offset]
        w = token_ids[: N - offset]
        c = token_ids[offset:]
        mask = (w >= 0) & (c >= 0)
        all_rows.append(w[mask])
        all_cols.append(c[mask])

        all_rows.append(c[mask])
        all_cols.append(w[mask])

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    data = np.ones(len(rows), dtype=np.float64)

    print("Building sparse matrix...")
    cooc = coo_matrix((data, (rows, cols)), shape=(V, V))
    return cooc.tocsr()
