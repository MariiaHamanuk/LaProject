# Word Embeddings via Matrix Factorization

Building word vector representations from scratch using truncated SVD applied to a PPMI-weighted co-occurrence matrix, and analyzing the geometric properties of the resulting embeddings.

**Authors:** Mariia Hamaniuk, Mykhailo Rykhalskyi, Ivan Zarytskyi

## Overview

This project demonstrates that meaningful semantic structure can be extracted from raw text using only counting and Linear Algebra. The method is theoretically grounded: optimality of the decomposition is guaranteed by the Eckart-Young theorem.

### Pipeline

1. Construct a co-occurrence matrix from the Text8 corpus (V=10,000, window size 5)
2. Apply PPMI weighting to remove frequency bias
3. Compute truncated SVD and extract k-dimensional word vectors
4. Evaluate via nearest-neighbour queries and word analogy tests
5. Compare embedding variants and analyze the singular value spectrum

### Evaluation

- **Qualitative:** nearest-neighbour checks and analogy arithmetic (e.g. king - man + woman ~ queen)
- **Quantitative:** Google Analogy Benchmark (19,544 questions), accuracy measured across k in {50, 100, 200, 300, 500}
- **Baselines:** raw co-occurrence cosine similarity, GloVe (300d), Word2Vec Skip-Gram (300d)

## Data

The project uses the **Text8** dataset (~680 MB) — 17 million tokens of cleaned Wikipedia text.

**Download:** https://www.kaggle.com/datasets/gupta24789/text8-word-embedding

Place the downloaded `text8` file into the `data/` directory:

```
data/
  text8
```

## Tech Stack

- Python
- `scipy.sparse` for memory-efficient matrix operations
- `scipy.sparse.linalg.svds` (Lanczos algorithm) for truncated SVD
