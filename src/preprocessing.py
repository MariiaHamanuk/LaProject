from collections import Counter
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm


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
    cooc = lil_matrix((V, V), dtype=np.float64)

    for i in tqdm(range(len(tokens)), desc="Building co-occurrence matrix"):
        word = tokens[i]
        if word not in word2idx:
            continue
        w_idx = word2idx[word]

        start = max(0, i - window)
        end = min(len(tokens), i + window + 1)

        for j in range(start, end):
            if j == i:
                continue
            context = tokens[j]
            if context not in word2idx:
                continue
            c_idx = word2idx[context]
            cooc[w_idx, c_idx] += 1

    return cooc.tocsr()
