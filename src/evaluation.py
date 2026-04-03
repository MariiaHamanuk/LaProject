import numpy as np
from collections import defaultdict


def normalize(E: np.ndarray) -> np.ndarray:
    """L2-normalize each row of E."""
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    return E / norms


def nearest_neighbours(
    word: str,
    E_norm: np.ndarray,
    idx2word: list,
    word2idx: dict,
    n: int = 10,
) -> list[tuple[str, float]]:
    """Find n nearest neighbours by cosine similarity.

    Args:
        E_norm: L2-normalized embedding matrix (V x k)

    Returns:
        List of (word, similarity) tuples, excluding the query word.
    """
    if word not in word2idx:
        return []
    idx = word2idx[word]
    scores = E_norm @ E_norm[idx]
    # Exclude the query word itself
    scores[idx] = -1
    top_indices = np.argsort(-scores)[:n]
    return [(idx2word[i], float(scores[i])) for i in top_indices]


def analogy(
    a: str,
    b: str,
    c: str,
    E_norm: np.ndarray,
    idx2word: list,
    word2idx: dict,
    n: int = 5,
) -> list[tuple[str, float]]:
    """Solve analogy: a is to b as c is to ?

    Computes: query = E[b] - E[a] + E[c], returns nearest neighbours.
    """
    for w in [a, b, c]:
        if w not in word2idx:
            return []

    query = E_norm[word2idx[b]] - E_norm[word2idx[a]] + E_norm[word2idx[c]]
    query = query / (np.linalg.norm(query) + 1e-10)

    scores = E_norm @ query
    # Exclude input words
    for w in [a, b, c]:
        scores[word2idx[w]] = -1

    top_indices = np.argsort(-scores)[:n]
    return [(idx2word[i], float(scores[i])) for i in top_indices]


def load_google_analogy(path: str) -> dict[str, list]:
    """Parse the Google Analogy Benchmark file.

    Format: lines starting with ':' are category headers,
    other lines are 'word_a word_b word_c word_d' (all lowercase).

    Returns:
        dict mapping category name -> list of (a, b, c, d) tuples
    """
    categories = defaultdict(list)
    current_category = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(":"):
                current_category = line[2:]
            else:
                parts = line.lower().split()
                if len(parts) == 4 and current_category:
                    categories[current_category].append(tuple(parts))

    return dict(categories)


def evaluate_analogy_benchmark(
    E_norm: np.ndarray,
    idx2word: list,
    word2idx: dict,
    categories: dict[str, list],
) -> dict:
    """Run the full Google Analogy Benchmark.

    Returns:
        dict with:
            "per_category": {category: {"correct": int, "total": int, "accuracy": float, "oov_skipped": int}}
            "overall_accuracy": float
            "total_correct": int
            "total_evaluated": int
            "total_oov_skipped": int
    """
    results = {}
    total_correct = 0
    total_evaluated = 0
    total_oov = 0

    for category, questions in categories.items():
        correct = 0
        evaluated = 0
        oov = 0

        for a, b, c, d in questions:
            # Skip if any word is out of vocabulary
            if any(w not in word2idx for w in [a, b, c, d]):
                oov += 1
                continue

            query = E_norm[word2idx[b]] - E_norm[word2idx[a]] + E_norm[word2idx[c]]
            query = query / (np.linalg.norm(query) + 1e-10)

            scores = E_norm @ query
            # Exclude input words
            for w in [a, b, c]:
                scores[word2idx[w]] = -1

            predicted_idx = np.argmax(scores)
            if idx2word[predicted_idx] == d:
                correct += 1
            evaluated += 1

        accuracy = correct / evaluated if evaluated > 0 else 0.0
        results[category] = {
            "correct": correct,
            "total": evaluated,
            "accuracy": accuracy,
            "oov_skipped": oov,
        }
        total_correct += correct
        total_evaluated += evaluated
        total_oov += oov

    overall_accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0

    return {
        "per_category": results,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_evaluated": total_evaluated,
        "total_oov_skipped": total_oov,
    }
