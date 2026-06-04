# Cross-Asset Portfolio RL — Set-Transformer Policy on a Long-Format Market Panel

A from-scratch reinforcement-learning system that allocates a **long-only,
cash-aware portfolio across a heterogeneous cross-asset universe** (crypto,
crypto miners, crypto-adjacent equities, payment networks, exchanges, banks,
data-center REITs, and semiconductors). The agent observes a per-asset feature
matrix each day and outputs per-asset target exposures; it is trained with PPO
(and an off-policy SAC variant) against a risk- and cost-aware reward.

This is **not** a wrapper around a stock RL library. The environment, the
look-ahead-safe data pipeline, the reward, the PnL accounting, and the policy
network were built specifically for the cross-sectional portfolio problem.

---

## TL;DR

- **Problem.** Given a daily panel of ~33 assets and several hundred features
  per asset, learn a daily long-only allocation that earns risk-adjusted return
  net of transaction costs, without look-ahead.
- **Approach.** A **cross-asset Set-Transformer** actor-critic: each asset is
  encoded independently, then assets attend to one another, then per-asset
  heads emit allocation signals. The design is permutation-equivariant over
  assets and supports a variable universe via attention masking.
- **Reward.** Log portfolio return plus rolling risk terms (Sharpe / Sortino /
  CVaR / max-drawdown), minus transaction costs and a penalty on infeasible
  ("blocked") trades, with an optional prospect-theory loss-aversion term.
- **Success criteria.** Out-of-sample (held-out time block) Sharpe and terminal
  NAV that beat two baselines: equal-weight rebalanced and buy-and-hold.
- **Honesty.** With a single historical price path, RL trading overfits easily.
  The controls used here, the residual risk, and the knobs to tighten are
  documented in [`OVERFITTING.md`](OVERFITTING.md).

---

## What is different from "usual" trading RL

Most published trading-RL setups flatten the market into a single state vector
and feed it to an MLP that emits one action per asset. That throws away the
cross-sectional structure (which assets are co-moving *right now*), fixes the
universe size, and tends to memorize a single price path. This repo differs on
five axes:

1. **Cross-asset attention policy (Set Transformer).** The observation is a
   matrix `(A assets × F features)`, not a flat vector. A shared encoder maps
   each asset's features to an embedding, cross-asset self-attention lets assets
   condition on each other, and per-asset heads produce a squashed-Gaussian
   action per asset. Because the asset axis is handled by attention + a learned
   asset-ID embedding, the policy is **permutation-equivariant** and can ingest
   a **variable / masked universe** (`key_padding_mask`) rather than a fixed
   flattened input. See `TransformerSquashedGaussianActorCritic.py` and the
   auto-selecting backend in `ppo_core_.py`.

2. **Bayesian trade-feasibility model.** `BlockProbabilityModel.py` maintains a
   per-asset **Beta-Bernoulli posterior** over the probability that an attempted
   trade gets blocked (min-notional, cash, non-negativity constraints). Its
   posterior mean is fed back into the observation, and a KL-based
   **information-gain** signal is exposed for curiosity-style shaping. This makes
   the agent aware of execution feasibility instead of pretending fills are free.

3. **Risk- and cost-shaped reward, not raw PnL.** `_compute_reward` combines log
   return with rolling risk metrics from `RewardShaper.py` /
   `PortfolioRiskFeatureEngine.py` (Sharpe, Sortino, CVaR, max-drawdown, beta-
   adjusted Sharpe), a transaction-cost penalty, a blocked-trade penalty, and an
   optional prospect-theory loss-aversion term. Weights are configurable in the
   env constructor (`reward_weights`).

4. **Look-ahead-safe data pipeline.** `env_starter.neutralization` splits by
   time, **fits the feature scaler on the training block only**, and preserves a
   raw, unscaled `Close_raw` execution price. Scaling the execution price could
   make it negative and produce phantom negative portfolio value ("fake
   leverage"); the pipeline explicitly forbids that and fails fast if execution
   prices go negative. The running observation scaler is fit during a random-
   action prefill and is dimension-checked on reload.

5. **Auditable execution + dual-basis PnL.** Every step writes a per-asset,
   per-timestamp ledger (`historical_trades_long`, MultiIndex `(timestamp, tic)`)
   and `LivePnLCalculator.py` tracks **FIFO and LIFO realized/unrealized PnL**,
   which are surfaced as observation features. The environment is reconstructable
   and inspectable after a run, not a black box.

---

## Architecture

```
                 LONG panel CSV  (timestamp, tic, ~588 features)
                            │
            env_starter.neutralization()   ← train-only scaler fit, Close_raw kept
                            │  train_df / val_df  (+ held-out test, see notes)
                            ▼
                    CryptoTradingEnv  (use_2d=True)
        per-asset obs matrix  (A × F)  =  market feats
                                          + position (qty, value, weight, cash)
                                          + last action (raw + softmax)
                                          + block-probability posterior
                                          + FIFO/LIFO PnL
                                          + portfolio risk features
                            │
                            ▼
        Set-Transformer Actor-Critic         reward = log_ret
        ├─ per-asset encoder  (F → D)                 + risk terms (Sharpe/Sortino/CVaR/MDD)
        ├─ cross-asset attention × N blocks           − tx_cost − blocked_trade
        ├─ actor heads → μ, logσ per asset            (+ optional prospect-theory)
        └─ critic: masked mean-pool → V
                            │
                            ▼
                 PPO (ppo.py)  /  SAC (sac.py)
```

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── sample_alpha_features.csv     # small runnable sample (6 assets, ~260 days)
├── OVERFITTING.md                # diagnosis + mitigation playbook
│
├── CryptoTradingEnv.py               # the environment (LONG / MultiIndex, 2D obs)
├── env_starter.py                    # look-ahead-safe split/scale + prefill
├── RewardShaper.py                   # rolling Sharpe/Sortino/CVaR/MDD
├── PortfolioRiskFeatureEngine.py     # portfolio risk features
├── LivePnLCalculator.py              # FIFO/LIFO realized/unrealized PnL
├── BlockProbabilityModel.py          # Beta-Bernoulli trade-feasibility tracker
│
├── TransformerSquashedGaussianActorCritic.py   # cross-asset Set Transformer
├── ppo.py  ppo_core_.py  ppo_buffer.py  ppo_mpi.py            # PPO stack
├── sac.py  sac_core.py  ReplayBuffers.py                       # SAC stack
├── base_agent.py  logger.py
│
├── run_train_ppo.py                  # end-to-end PPO runner (primary)
├── run_train_sac.py                  # end-to-end SAC runner
├── tune_ppo.py  tune_sac.py          # hyperparameter sweeps
├── ppo_train_test_plot.py  sac_train_test_plot.py
└── test_integration_long_env.py      # wiring / correctness smoke test
```

> **Note on duplicates.** Several files ship in both a current and a legacy form
> (e.g. `CryptoTradingEnv.py` vs `CryptoTradingEnv_.py`, `env_starter.py` vs
> `_env_starter.py`, `tune_ppo.py` vs `tune_ppo_.py`). The current files are the
> ones referenced by the runners above. Legacy `_`-suffixed/prefixed files are
> kept only for diffing and should be deleted before a clean submission.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
```

`gym`/`gymnasium`, `tqdm`, `plotext`, and `mpi4py` are optional — the code falls
back gracefully when they are absent (MPI degrades to single-process, plots are
skipped). Install `mpi4py` (and a system MPI such as OpenMPI) only if you want
multi-process PPO rollouts.

---

## Data

The environment expects **long-format** data: columns `timestamp`, `tic`, a
`Close` (and ideally `Close_raw`) execution price, and any number of feature
columns. Rows are one `(timestamp, tic)` observation.

- **Runnable sample:** `sample_alpha_features.csv` — 6 assets
  (BTC, ETH, MARA, RIOT, NVDA, EQIX), ~260 daily bars, all feature columns. Use
  it to verify the pipeline end to end.
- **Full dataset:** the full panel (33 assets, 2020–2026, ~588 features) is
  ~386 MB and is **not** committed (it exceeds GitHub's 100 MB file limit). It is
  produced by an external feature pipeline (TA-Lib technical indicators + custom
  alpha factors + index/risk composites). Point any runner at it with `--data`.

---

## Quickstart

```bash
# 1) Smoke test the environment wiring on the sample
python test_integration_long_env.py --data sample_alpha_features.csv --steps 200

# 2) Train PPO with the Set-Transformer policy (auto-selected for 2D obs)
python run_train_ppo.py \
    --data sample_alpha_features.csv \
    --use-2d --procs 1 --train-epochs 10 --steps-per-epoch 2048

python run_train_ppo.py \
    --data sample_alpha_features.csv \
    --procs 1 --use-2d --train-epochs 4000 --test-episodes 2 --horizon 0

# 3) Train SAC (MLP-over-flattened-obs baseline)
python run_train_sac.py --data sample_alpha_features.csv --use-2d --epochs 10
```

The PPO runner is the maintained entry point. The Set-Transformer backend is
selected automatically when the observation is 2D `(A, F)` and `act_dim == A`.

---

## Success criteria & evaluation

Training reward is a means, not the metric. Success is defined **out of sample**
on a time block the agent never trained or tuned on:

- **Primary:** out-of-sample annualized Sharpe > both baselines.
- **Secondary:** terminal NAV and max-drawdown vs the same baselines.
- **Baselines:** (a) equal-weight, periodically rebalanced; (b) buy-and-hold the
  initial equal-weight basket.
- **Sanity:** turnover and net-of-cost return — a strategy that only wins gross
  of transaction costs is not a strategy.

A run is considered a *failure* (overfit) when in-sample reward rises while the
out-of-sample Sharpe stagnates or falls. That gap is the headline diagnostic in
[`OVERFITTING.md`](OVERFITTING.md).

---

## Known limitations

- **Single price path.** The model is trained on one realized history. This is
  the dominant overfitting risk; see the mitigation playbook.
- **Test split.** `neutralization` currently returns train/val and *drops* the
  final test block to preserve a legacy return signature. Re-enable a true
  held-out test before reporting any out-of-sample number (one-line fix noted in
  `OVERFITTING.md`).
- **Daily bars only.** No intraday microstructure; execution is modeled at the
  daily close with a proportional + fixed fee and a min-notional constraint.
- **SAC runner.** `run_train_sac.py` passes an `agent=` kwarg that
  `prefill_replay_buffer_and_scalers` does not accept; remove it (the PPO runner
  already does) before running SAC.

---

## License

MIT — see [`LICENSE`](LICENSE).
