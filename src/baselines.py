import numpy as np
from scipy.sparse import csr_matrix
from src.evaluation import normalize
from gensim.models import KeyedVectors

def cosine_baseline_vectors(cooccurrence: csr_matrix) -> np.ndarray:
    """
    Creates baseline word vectors from raw co-occurrence counts.
    Returns dense normalized matrix for use with evaluation functions.
    """
    return normalize(cooccurrence.toarray().astype(np.float64))


def load_glove(path: str, word2idx: dict, dim: int = 300) -> np.ndarray:
    """
    Load pretrained GloVe vectors and align to our vocabulary.
    Args:
        path: to GloVe text file
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

    print(f"GloVe is loaded {found}/{V} words")
    return E


def load_word2vec(path: str, word2idx: dict) -> np.ndarray:
    """
    Load pretrained Word2Vec vectors and align to our vocabulary.
    Args:
        path: path to Word2Vec
        word2idx: vocabulary mapping
    Returns:
        E: (V x dim) matrix, rows aligned to word2idx. Words not found get zeros.
    """
    wv = KeyedVectors.load_word2vec_format(path, binary=True)
    dim = wv.vector_size
    V = len(word2idx)
    E = np.zeros((V, dim), dtype=np.float64)
    found = 0
    for word, idx in word2idx.items():
        if word in wv:
            E[idx] = wv[word]
            found += 1
    print(f"Word2Vec is loaded {found}/{V} words")
    return E
