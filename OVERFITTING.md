# Overfitting: diagnosis and mitigation

RL trading on a single historical price path is one of the easiest places in
all of ML to fool yourself. This document explains *why* this system overfits
and gives a prioritized, code-anchored playbook to fix it.

## Why it overfits (the numbers)

On the full panel, `neutralization(start_step_train=0, train_ratio=0.7)` yields:

| Split | Timesteps | Dates |
|-------|-----------|-------|
| Train | **907**   | 2020-12-23 → 2024-08-02 |
| Val   | 194       | 2024-08-05 → 2025-05-13 |
| Test  | 196       | *computed then dropped* |

Against those **907 training days** the agent sees:

- **~588 features per asset × 33 assets ≈ 19,400-dimensional** flattened input
  (for SAC's MLP), or 33 attention tokens of ~601 features each (for the
  Set-Transformer).
- A **single realized trajectory**. Episodes differ only by random start index,
  so every episode replays slices of the *same* path. There is no second world
  in which BTC didn't do what it did in 2021–2024.

High-capacity function approximators + one path + hundreds of correlated
features = memorization. The agent learns *this history*, not a *policy*.

The tell: **in-sample episodic return keeps climbing while validation Sharpe
flattens or decays.** Plot both every epoch; the gap is the diagnosis.

---

## The fixes, in order of leverage

### 1. Train on more than one path (highest leverage)

The single-path problem dominates everything else. Two practical options:

- **Walk-forward / rolling-origin training.** Instead of one 70/15/15 split,
  train on window *k*, validate on *k+1*, roll forward, and report the
  *concatenated out-of-sample* curve. This is the standard defense in
  quantitative finance and directly measures generalization across regimes.
- **Block bootstrap of the return path.** Resample contiguous blocks of the
  joint (cross-asset) return series to synthesize many plausible alternative
  histories, and start episodes from those. Blocks (not i.i.d. days) preserve
  autocorrelation and cross-asset correlation. Even 50–100 synthetic paths
  changes the problem from "memorize one history" to "learn what generalizes."

Either one is worth more than every other item on this list combined.

### 2. Restore a true held-out test split

`env_starter.neutralization` builds `test_df` and then throws it away to keep a
legacy 6-tuple return order. Re-enable it and **never tune on it**:

```python
# in neutralization(...), instead of returning train_df, val_df, ...
return train_df, val_df, test_df, train_price, val_price, train_scaler, test_scaler
```

Tune on validation, report once on test at the very end. A number you tuned
against is an in-sample number wearing a costume.

### 3. Add purge + embargo to any cross-validation

Adjacent train/val days share overlapping rolling-window features (the env's
`window_size=30`), so a naive split leaks. Drop (purge) the days whose feature
windows straddle the train/val boundary and add a small embargo gap after it.
This is López de Prado's purged K-fold; it matters here because almost every
feature is a trailing-window statistic.

### 4. Cut capacity and turn on regularization

Defaults ship with **no dropout and no weight decay** — fine for debugging,
wrong for a 907-sample problem.

- **Dropout.** `ppo_core_.MLPActorCritic(..., st_dropout=0.1)` (or `0.2`). The
  Set-Transformer blocks already thread dropout through; it just defaults to 0.
- **Weight decay.** Use `AdamW` (or `Adam(..., weight_decay=1e-4)`) for both
  actor and critic optimizers in `ppo.py` / `sac.py`.
- **Smaller network.** Drop `st_d_model` 128 → 64 and `st_n_blocks` 2 → 1. With
  907 samples, a smaller policy generalizes better than a bigger one.

### 5. Reduce the feature count

588 features per asset for 907 days is an adverse ratio. Most TA-Lib indicators
are near-duplicates of a handful of underlying signals. Options:

- Curate a **small, economically-motivated subset** (e.g. 20–40 features:
  returns at a few horizons, a volatility measure, a volume/liquidity measure, a
  couple of cross-sectional ranks) and drop the rest.
- Or fit **PCA on the training block only** and keep enough components for ~95%
  variance. Fit on train, transform val/test — same hygiene as the scaler.

Fewer, decorrelated inputs shrink the hypothesis space the agent can overfit to.

### 6. Early-stop on validation, not on training reward

You already construct a `test_env` on the validation block. Use it: evaluate
validation Sharpe/return every *N* epochs, keep the best checkpoint, and stop
when validation hasn't improved for a patience window. Without this, longer
training simply means more memorization.

### 7. Make costs and exploration bite

- **Transaction costs.** If `transaction_fee` / `tx_fee_w` are too small, the
  agent can churn to fit noise for free. Set them at or above realistic levels;
  a strategy that survives realistic costs is far less likely to be curve-fit.
- **PPO entropy bonus.** Keep a non-trivial entropy coefficient so the policy
  doesn't collapse onto the one in-sample-optimal sequence of weights.
- **Shorter horizons / more resets.** More frequent episode resets from varied
  start points reduce reliance on any single long stretch of history.

---

## A minimal "is it still overfitting?" checklist

1. Plot in-sample vs out-of-sample (val) Sharpe per epoch. Diverging? Overfit.
2. Does the policy beat **equal-weight** and **buy-and-hold** *out of sample*,
   *net of costs*? If not, there is no edge to deploy.
3. Shuffle the asset order at evaluation. A permutation-equivariant policy
   should be invariant; a large change means it latched onto asset identity.
4. Re-run with a different seed and a different walk-forward window. If results
   swing wildly, the "performance" is sampling luck.

If items 1–4 hold up, you have something worth writing about. Until then, the
training curve is just a memorization curve.
