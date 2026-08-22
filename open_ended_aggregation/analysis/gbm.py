"""
Histogram gradient boosting, logistic loss. ~200 lines, no new dependencies.

WHY IT IS HERE
  The scoring model in weighting_v3/v4 is a logistic regression. The signals it
  is being asked to combine are strongly interactive: rank matters much more for
  a verbose agent than a terse one; a solo claim from a short list means
  something different from a solo claim from a 2431-item list; the value of a
  support count depends on WHICH agents and where in their lists. A linear model
  in these features cannot express any of that, and the diagnostic in v4 says
  the ordering -- not the keep-rule -- is what is now limiting:

    LEAN + budget    31.71      the best linear-scored method
    ORACLE selection 59.34      keep exactly the correct claims

  sklearn is not installed and the project has deliberately stayed dependency
  -light, so this is a compact histogram GBM: features pre-binned once into
  uint8 codes, then per node one bincount per feature gives the split gain in
  closed form. Cost per tree is O(depth x features x rows).

Not general-purpose -- it does exactly what this analysis needs and nothing else.
"""
import numpy as np


def bin_features(X, nbins=32):
    """Quantile-bin every column to uint8 codes. Returns codes and edges."""
    codes = np.zeros(X.shape, dtype=np.uint8)
    edges = []
    for j in range(X.shape[1]):
        col = X[:, j]
        u = np.unique(col)
        if len(u) <= nbins:
            e = u[:-1] if len(u) > 1 else np.array([-np.inf])
        else:
            e = np.unique(np.quantile(col, np.linspace(0, 1, nbins + 1)[1:-1]))
        codes[:, j] = np.searchsorted(e, col, side="right")
        edges.append(e)
    return codes, edges


def apply_bins(X, edges):
    codes = np.zeros(X.shape, dtype=np.uint8)
    for j, e in enumerate(edges):
        codes[:, j] = np.searchsorted(e, X[:, j], side="right")
    return codes


def _best_split(sub, g, h, idx, nb, lam, min_h, min_gain):
    G, H = g[idx].sum(), h[idx].sum()
    parent = G * G / (H + lam)
    best = None
    for j in range(sub.shape[1]):
        c = sub[:, j]
        hg = np.bincount(c, weights=g[idx], minlength=nb)
        hh = np.bincount(c, weights=h[idx], minlength=nb)
        GL = np.cumsum(hg)[:-1]
        HL = np.cumsum(hh)[:-1]
        GR, HR = G - GL, H - HL
        ok = (HL >= min_h) & (HR >= min_h)
        if not ok.any():
            continue
        gain = GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent
        gain = np.where(ok, gain, -np.inf)
        b = int(np.argmax(gain))
        if gain[b] > (best[0] if best else min_gain):
            best = (float(gain[b]), j, b)
    return best


def _grow(codes, g, h, idx, depth, max_depth, nb, lam, min_h, min_gain):
    if depth == max_depth or len(idx) < 2 * min_h:
        return {"leaf": -g[idx].sum() / (h[idx].sum() + lam)}
    sub = codes[idx]
    sp = _best_split(sub, g, h, idx, nb, lam, min_h, min_gain)
    if sp is None:
        return {"leaf": -g[idx].sum() / (h[idx].sum() + lam)}
    _, j, b = sp
    m = sub[:, j] <= b
    li, ri = idx[m], idx[~m]
    if not len(li) or not len(ri):
        return {"leaf": -g[idx].sum() / (h[idx].sum() + lam)}
    return {"j": j, "b": b,
            "L": _grow(codes, g, h, li, depth + 1, max_depth, nb, lam, min_h, min_gain),
            "R": _grow(codes, g, h, ri, depth + 1, max_depth, nb, lam, min_h, min_gain)}


def _predict_tree(node, codes, out, idx):
    if "leaf" in node:
        out[idx] += node["leaf"]
        return
    m = codes[idx, node["j"]] <= node["b"]
    _predict_tree(node["L"], codes, out, idx[m])
    _predict_tree(node["R"], codes, out, idx[~m])


class GBM:
    def __init__(self, n_trees=200, lr=0.15, max_depth=4, nbins=32,
                 lam=1.0, min_h=20.0, min_gain=1e-6, subsample=1.0, seed=0):
        self.n_trees, self.lr, self.max_depth = n_trees, lr, max_depth
        self.nbins, self.lam, self.min_h, self.min_gain = nbins, lam, min_h, min_gain
        self.subsample, self.seed = subsample, seed
        self.trees, self.base, self.edges = [], 0.0, None

    def fit(self, X, y, codes=None, edges=None, val=None, patience=25):
        """val = (Xv_codes, yv) for early stopping on held-out log loss."""
        if codes is None:
            codes, edges = bin_features(X, self.nbins)
        self.edges = edges
        p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.base = float(np.log(p0 / (1 - p0)))
        F = np.full(len(y), self.base)
        Fv = None
        if val is not None:
            Fv = np.full(len(val[1]), self.base)
        rng = np.random.default_rng(self.seed)
        best, bad, keep = np.inf, 0, 0
        for t in range(self.n_trees):
            p = 1.0 / (1.0 + np.exp(-np.clip(F, -30, 30)))
            g, h = p - y, np.clip(p * (1 - p), 1e-6, None)
            idx = np.arange(len(y))
            if self.subsample < 1.0:
                idx = idx[rng.random(len(y)) < self.subsample]
            tree = _grow(codes, g, h, idx, 0, self.max_depth,
                         self.nbins + 1, self.lam, self.min_h, self.min_gain)
            self.trees.append(tree)
            step = np.zeros(len(y))
            _predict_tree(tree, codes, step, np.arange(len(y)))
            F += self.lr * step
            if val is not None:
                sv = np.zeros(len(val[1]))
                _predict_tree(tree, val[0], sv, np.arange(len(val[1])))
                Fv += self.lr * sv
                pv = 1.0 / (1.0 + np.exp(-np.clip(Fv, -30, 30)))
                ll = -np.mean(val[1] * np.log(np.clip(pv, 1e-9, 1)) +
                              (1 - val[1]) * np.log(np.clip(1 - pv, 1e-9, 1)))
                if ll < best - 1e-6:
                    best, bad, keep = ll, 0, t + 1
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if val is not None and keep:
            self.trees = self.trees[:keep]
        return self

    def decision(self, codes):
        out = np.full(len(codes), self.base)
        idx = np.arange(len(codes))
        for tree in self.trees:
            step = np.zeros(len(codes))
            _predict_tree(tree, codes, step, idx)
            out += self.lr * step
        return out
