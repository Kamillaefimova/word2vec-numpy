# word2vec-numpy
word2vec algo code

# Word2Vec from Scratch — NumPy Implementation

Implementation of the **Skip-Gram with Negative Sampling (SGNS)** variant of Word2Vec
using only NumPy — no PyTorch, TensorFlow, or any other ML framework.

Built as part of an application for a research internship on efficient language model training.

---

## What is Word2Vec?

Word2Vec learns dense vector representations of words by training a shallow neural network
to predict context words from a center word (Skip-Gram).

Words that appear in similar contexts end up with similar vectors — so the geometry
of the embedding space captures semantic relationships.

---

## This Implementation

**Model:** Skip-Gram with Negative Sampling (Mikolov et al., 2013)

For each (center, context) pair, the loss is:
```
L = -log σ(v_c · u_pos) - Σ log σ(-v_c · u_neg_k)
```

where σ is the sigmoid function, `v_c` is the center word vector,
`u_pos` is the context vector, and `u_neg_k` are K noise word vectors.

**Gradients (derived by hand):**
```
δ_pos = σ(v_c · u_pos) - 1

∂L/∂u_pos   = δ_pos · v_c
∂L/∂u_neg_k = σ(v_c · u_neg_k) · v_c
∂L/∂v_c     = δ_pos · u_pos + Σ_k σ(v_c · u_neg_k) · u_neg_k
```

Updates use vanilla SGD: `θ ← θ - lr · ∇θ`

---

## Project Structure
```
word2vec-numpy/
│
├── word2vec_numpy.py   # full implementation
└── README.md
```

**`word2vec_numpy.py` contains:**
| Function / Class | Role |
|---|---|
| `load_corpus()` | loads and tokenizes text |
| `build_vocab()` | maps words ↔ integer indices |
| `noise_distribution()` | unigram^(3/4) noise distribution for negative sampling |
| `build_skipgram_pairs()` | generates (center, context) training pairs |
| `Word2Vec` | model: embeddings, forward pass, gradients, SGD update |
| `Word2Vec.most_similar()` | cosine nearest-neighbour lookup |

---

## How to Run

**Requirements:** Python 3.8+, NumPy
```bash
pip install numpy
python word2vec_numpy.py
```

By default the script runs on a built-in sample corpus.
To use your own dataset, set `CORPUS_PATH` in `main()`:
```python
CORPUS_PATH = "path/to/your/text.txt"   # any plain-text file
```

**Hyperparameters** (configurable at the top of `main()`):

| Parameter | Default | Description |
|---|---|---|
| `EMBED_DIM` | 64 | embedding dimension |
| `WINDOW` | 2 | context window size |
| `NEG_SAMPLES` | 5 | negative samples per pair |
| `EPOCHS` | 10 | training epochs |
| `LR` | 0.025 | SGD learning rate |

---

## Example Output
```
Vocab size: 148   Tokens: 219
Training pairs: 870
Training  V=148  D=64  pairs=870  epochs=10  K=5  lr=0.025

Epoch  1/10  avg_loss=4.1578
Epoch  5/10  avg_loss=4.0981
Epoch 10/10  avg_loss=3.4207

Nearest neighbours:
      language → models (0.864), word (0.833), processing (0.812)
         model → trained (0.794), word (0.788), language (0.761)
    embeddings → word (0.841), depth (0.802), representations (0.791)
```

Loss decreases consistently across epochs, confirming correct gradient computation.

---

## Key Design Decisions

**Two embedding matrices** — `W_in` (center) and `W_out` (context) are kept separate,
exactly as in the original paper. Only `W_in` is used as the final word representation.

**Noise distribution** — unigram frequencies raised to the ¾ power so that rare words
are sampled more often than their raw frequency would suggest.

**Numerical stability** — sigmoid inputs are clipped to [-30, 30] and a small ε is added
inside log() to prevent log(0).

**Xavier initialisation** — `W_in` is initialised with uniform noise scaled by `1/√D`
to keep gradient magnitudes stable early in training. `W_out` starts at zero.

---

## Reference

Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013).
*Distributed Representations of Words and Phrases and their Compositionality.*
NeurIPS 2013. https://arxiv.org/abs/1310.4546
