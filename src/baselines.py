import numpy as np
from scipy.sparse import csr_matrix
from src.evaluation import normalize


def cosine_baseline_vectors(cooccurrence: csr_matrix) -> np.ndarray:
    """Create baseline word vectors from raw co-occurrence counts.

    Simply L2-normalizes each row of the co-occurrence matrix.
    Returns dense normalized matrix for use with evaluation functions.
    """
    dense = cooccurrence.toarray().astype(np.float64)
    return normalize(dense)


def load_glove(path: str, word2idx: dict, dim: int = 300) -> np.ndarray:
    """Load pretrained GloVe vectors and align to our vocabulary.

    Args:
        path: path to GloVe text file (e.g. glove.6B.300d.txt)
        word2idx: our vocabulary mapping
        dim: embedding dimension

    Returns:
        E: (V x dim) matrix, rows aligned to word2idx. Words not in GloVe get zeros.
    """
    V = len(word2idx)
    E = np.zeros((V, dim), dtype=np.float64)
    found = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in word2idx:
                E[word2idx[word]] = np.array(parts[1:], dtype=np.float64)
                found += 1

    print(f"GloVe: loaded {found}/{V} words")
    return E


def load_word2vec(path: str, word2idx: dict) -> np.ndarray:
    """Load pretrained Word2Vec vectors (via gensim) and align to our vocabulary.

    Args:
        path: path to Word2Vec binary file (e.g. GoogleNews-vectors-negative300.bin)
        word2idx: our vocabulary mapping

    Returns:
        E: (V x dim) matrix, rows aligned to word2idx. Words not found get zeros.
    """
    from gensim.models import KeyedVectors

    wv = KeyedVectors.load_word2vec_format(path, binary=True)
    dim = wv.vector_size
    V = len(word2idx)
    E = np.zeros((V, dim), dtype=np.float64)
    found = 0

    for word, idx in word2idx.items():
        if word in wv:
            E[idx] = wv[word]
            found += 1

    print(f"Word2Vec: loaded {found}/{V} words")
    return E
