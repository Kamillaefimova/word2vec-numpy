import re
import time
import collections
import numpy as np

RNG = np.random.default_rng(42)

BUILTIN_TEXT = """
Natural language processing is a subfield of linguistics and artificial intelligence
concerned with the interactions between computers and human language.
Deep learning has revolutionized the way machines understand and generate text.
Word embeddings such as word2vec learn dense vector representations of words
by predicting surrounding context words from a center word.
The skip-gram model with negative sampling is an efficient way to train such embeddings.
Recurrent neural networks process sequences step by step maintaining hidden states.
Transformer architectures use self-attention to relate all positions in a sequence.
Language models assign probabilities to sequences of words.
Small language models can still achieve strong performance on many tasks.
Computational budgets constrain the size and depth of models that can be trained.
Reasoning capabilities emerge from sufficient model depth and training data.
Masked language models like BERT are pre-trained on large text corpora.
Fine-tuning adapts a pre-trained model to a specific downstream task.
Negative sampling approximates the full softmax by sampling noise words.
The noise distribution is typically the unigram frequency raised to the 3/4 power.
Gradient descent iteratively updates parameters to minimize the loss function.
Backpropagation computes gradients efficiently via the chain rule.
Embeddings capture semantic relationships so that similar words cluster together.
Training word vectors on large corpora reveals analogical structure in the embedding space.
"""


def load_corpus(path: str | None = None) -> list[str]:
    if path:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = BUILTIN_TEXT
    tokens = re.findall(r"[a-z]+", raw.lower())
    return tokens


def build_vocab(tokens: list[str], min_count: int = 1):
    freq = collections.Counter(tokens)
    vocab = [w for w, c in freq.most_common() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = vocab
    counts = np.array([freq[w] for w in vocab], dtype=np.float64)
    return word2idx, idx2word, counts


def noise_distribution(counts: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    p = counts ** alpha
    return p / p.sum()


def build_skipgram_pairs(tokens, word2idx, window: int = 2):
    ids = [word2idx[t] for t in tokens if t in word2idx]
    pairs = []
    for i, center in enumerate(ids):
        lo = max(0, i - window)
        hi = min(len(ids), i + window + 1)
        for j in range(lo, hi):
            if j != i:
                pairs.append((center, ids[j]))
    return pairs


class Word2Vec:

    def __init__(self, vocab_size: int, embed_dim: int = 64):
        self.V = vocab_size
        self.D = embed_dim
        scale = 1.0 / np.sqrt(embed_dim)
        self.W_in  = RNG.uniform(-scale, scale, (vocab_size, embed_dim))  # center
        self.W_out = np.zeros((vocab_size, embed_dim))                    # context


    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _neg_sample(self, noise_probs: np.ndarray, k: int,
                    exclude: tuple) -> np.ndarray:
        samples = []
        while len(samples) < k:
            cands = RNG.choice(self.V, size=k * 2, p=noise_probs)
            for c in cands:
                if c not in exclude:
                    samples.append(c)
                if len(samples) == k:
                    break
        return np.array(samples[:k])


    def train_pair(self, center: int, context: int,
                   noise_probs: np.ndarray, k: int,
                   lr: float) -> float:

        neg_ids = self._neg_sample(noise_probs, k, exclude=(center, context))

        v_c   = self.W_in[center]           # (D,)
        u_pos = self.W_out[context]         # (D,)
        u_neg = self.W_out[neg_ids]         # (K, D)

        # ── forward ──────────────────────────────────────────────────────
        s_pos = v_c @ u_pos                 # scalar
        s_neg = u_neg @ v_c                 # (K,)

        sig_pos = self._sigmoid( s_pos)     # scalar
        sig_neg = self._sigmoid( s_neg)     # (K,)

        loss = (-np.log(sig_pos + 1e-10)
                - np.sum(np.log(1.0 - sig_neg + 1e-10)))

        # ── backward ─────────────────────────────────────────────────────
        delta_pos = sig_pos - 1.0                        # scalar
        delta_neg = sig_neg                              # (K,)

        grad_u_pos = delta_pos * v_c                     # (D,)
        grad_u_neg = delta_neg[:, None] * v_c[None, :]   # (K, D)
        grad_v_c   = (delta_pos * u_pos
                      + delta_neg @ u_neg)               # (D,)

        # ── SGD updates ──────────────────────────────────────────────────
        self.W_out[context]  -= lr * grad_u_pos
        self.W_out[neg_ids]  -= lr * grad_u_neg
        self.W_in [center]   -= lr * grad_v_c

        return float(loss)

    # ── training loop ────────────────────────────────────────────────────

    def fit(self, pairs: list, noise_probs: np.ndarray,
            epochs: int = 5, k: int = 5, lr: float = 0.025,
            log_every: int = 5000):
        """Iterate over (center, context) pairs for *epochs* epochs."""
        n = len(pairs)
        print(f"Training  V={self.V}  D={self.D}  "
              f"pairs={n}  epochs={epochs}  K={k}  lr={lr}")
        for epoch in range(1, epochs + 1):
            # shuffle pairs each epoch
            order = RNG.permutation(n)
            total_loss = 0.0
            t0 = time.time()
            for step, idx in enumerate(order):
                c, ctx = pairs[idx]
                total_loss += self.train_pair(c, ctx, noise_probs, k, lr)
                if log_every and (step + 1) % log_every == 0:
                    avg = total_loss / (step + 1)
                    elapsed = time.time() - t0
                    print(f"  epoch {epoch}  step {step+1:>6}/{n}"
                          f"  avg_loss={avg:.4f}  {elapsed:.1f}s")
            avg = total_loss / n
            print(f"Epoch {epoch}/{epochs}  avg_loss={avg:.4f}"
                  f"  time={time.time()-t0:.1f}s")


    def get_embedding(self, word: str, word2idx: dict) -> np.ndarray:
        idx = word2idx[word]
        v = self.W_in[idx]
        return v / (np.linalg.norm(v) + 1e-10)

    def most_similar(self, word: str, word2idx: dict,
                     idx2word: list, topn: int = 5):
        if word not in word2idx:
            return []
        q = self.get_embedding(word, word2idx)          # (D,)
        norms = np.linalg.norm(self.W_in, axis=1) + 1e-10
        sims  = self.W_in @ q / norms                   # (V,)
        top   = np.argsort(-sims)
        results = []
        for i in top:
            if idx2word[i] != word:
                results.append((idx2word[i], float(sims[i])))
            if len(results) == topn:
                break
        return results


def main():
    # ── config ──────────────────────────────────────────────────────────
    CORPUS_PATH = None       # set to a .txt file path to use your own data
    WINDOW      = 2
    EMBED_DIM   = 64
    EPOCHS      = 10
    NEG_SAMPLES = 5
    LR          = 0.025
    MIN_COUNT   = 1
    LOG_EVERY   = 0          # set to e.g. 5000 to print mid-epoch stats

    print("Loading corpus …")
    tokens = load_corpus(CORPUS_PATH)
    word2idx, idx2word, counts = build_vocab(tokens, min_count=MIN_COUNT)
    V = len(word2idx)
    print(f"Vocab size: {V}   Tokens: {len(tokens)}")

    noise_probs = noise_distribution(counts)

    print("Building skip-gram pairs …")
    pairs = build_skipgram_pairs(tokens, word2idx, window=WINDOW)
    print(f"Training pairs: {len(pairs)}")

    # ── train ───────────────────────────────────────────────────────────
    model = Word2Vec(vocab_size=V, embed_dim=EMBED_DIM)
    model.fit(pairs, noise_probs,
              epochs=EPOCHS, k=NEG_SAMPLES,
              lr=LR, log_every=LOG_EVERY)

    # ── demo: nearest neighbours ─────────────────────────────────────────
    print("\nNearest neighbours (cosine similarity in W_in space):")
    probe_words = ["language", "model", "training", "embeddings", "learning"]
    for w in probe_words:
        if w in word2idx:
            nbrs = model.most_similar(w, word2idx, idx2word, topn=5)
            nbr_str = ", ".join(f"{ww} ({s:.3f})" for ww, s in nbrs)
            print(f"  {w:>12s} → {nbr_str}")

    # ── save embeddings ──────────────────────────────────────────────────
    out_path = "embeddings.txt"
    with open(out_path, "w") as f:
        f.write(f"{V} {EMBED_DIM}\n")
        for i, word in enumerate(idx2word):
            vec_str = " ".join(f"{x:.6f}" for x in model.W_in[i])
            f.write(f"{word} {vec_str}\n")
    print(f"\nEmbeddings saved → {out_path}")


if __name__ == "__main__":
    main()
