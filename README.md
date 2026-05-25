# Gemini Quant Challenge — Market-Making (3rd place)

Team **Moon Shot** (Columbia MSFE) — 3rd out of 50 teams at the *Gemini Collegiate Quant Trading Competition* (Feb 15 – Apr 15, 2026). $5,000 prize.

## Context

The competition ran on the **live Gemini exchange** for two months. Each team received a $3,000 cash advance (Gemini absorbed losses) and was ranked on:

```
Daily Score = Volume × (1 + ROE)²       Total Score = Σ over the 5 best days
```

where ROE = Ending Balance / Beginning Balance. The formula incentivises *high notional turnover under a no-loss constraint* — i.e. market-making, not directional alpha.

**The pivot.** Our initial edge was a passive stablecoin maker (USDC/USD + USDT/USD), built around the GTQB strategy in [`mm/gtqb_usdc_usdt.py`](mm/gtqb_usdc_usdt.py). Two structural reasons made stablecoins the obvious target under a `Volume × (1 + ROE)²` scoring rule:

1. **Zero trading fees** on stablecoin pairs on Gemini — every executed dollar of notional counted toward Volume with no drag on ROE.
2. **Pegged price** — inventory risk is essentially flat, so we could run very high turnover without exposing the book to MTM swings that would have crushed ROE on volatile pairs.

**Three weeks before the end of the competition, the organisers ruled that stablecoin volume would no longer count toward scoring.** Our primary book was suddenly worth zero points. We pivoted to volatile pairs (XRP/USD, BNB/USD, SOL/USD) using the Avellaneda-Stoikov implementation, re-tuned every parameter through the Bayesian optimiser, and kept the live infrastructure stable throughout. The fact that the [data collector](mm/mmbt/data/collect.py) had been streaming all the non-stablecoin pairs to disk *in parallel since day one* — even though we weren't trading them — meant we had clean order-book history ready for re-optimisation the moment we needed it. From there we **climbed from last place to 3rd in the final stretch**. The modularity that let us swap an entire strategy + asset universe in a few days is the part we are most proud of — and it is the main reason this codebase looks the way it does.

## Approach

**Two halves of the stack:**

- [`mm/`](mm/) — a pure-Python research framework (**mmbt**) on top of [hftbacktest](https://github.com/nkaz001/hftbacktest), used for strategy R&D and parameter tuning. Three strategies live here:
  - **Avellaneda-Stoikov** ([algorithms/as_algo.py](mm/mmbt/algorithms/as_algo.py)) — reservation price `r = s − qγσ²τ + α`, optimal spread `γσ²τ + (2/γ)ln(1 + γ/k)`, with online σ estimation, optional auto-calibration of `k` from message rate, and pluggable alpha signals.
  - **GTQB** ([algorithms/gtqb_algo.py](mm/mmbt/algorithms/gtqb_algo.py)) — a hybrid Grid Trading + Queue-Based market maker. The QB component quotes at BBO and steps back one tick when the queue is thin in the direction of inventory; the GT component layers limit orders around a skew-adjusted reservation price.
  - **OBI** ([algorithms/obi_algo.py](mm/mmbt/algorithms/obi_algo.py)) — order-book-imbalance quoter, also exposed as a pluggable [`OBISignal`](mm/mmbt/algorithms/signals/obi_signal.py) that can be added to any other strategy.

- [`qb/`](qb/) — the live trading bot we actually ran during the competition. It connects to Gemini via REST + WebSocket ([exchanges.py](qb/exchanges.py)), holds an in-memory L2 book via a [C++ tracker](qb/src/tracker.cpp) compiled to a Python module, and runs the same three strategies in production. A separate analytics dashboard ([dashboard.py](qb/dashboard.py)) subscribes to a ZMQ event stream and serves real-time metrics over a Dash UI.

**Tooling that mattered:**

- A **Bayesian optimiser** ([optimizer/bayesian.py](mm/mmbt/optimizer/bayesian.py)) wraps scikit-optimize and supports three parameter layouts (scalar, fixed list, per-asset range) plus walk-forward / k-fold CV out of the box ([optimizer/cv.py](mm/mmbt/optimizer/cv.py)). We used it with the competition's actual scoring formula as the objective.
- A **live data collector** ([data/collect.py](mm/mmbt/data/collect.py)) parses Gemini's and Binance's WS streams and writes hourly NPZ files in hftbacktest's exact format, with periodic book snapshots so backtests can warm-start cleanly at any hour.

**The objective, restated as an optimisation problem.** Once we lost stablecoin scoring and had to quote on volatile pairs with non-zero maker fees, the strategy stopped being "market-make and pocket the spread" and became *"buy as much scored volume as possible from a finite fee budget"*. Concretely: let `L_i` be the dollar loss (mostly fees + adverse selection) on day `i`. Beginning balance is the cash advance `C = $3,000`, so

```
ROE_i = (C − L_i) / C            (1 + ROE_i)² = (2 − L_i/C)²
Daily Score_i = V_i × (2 − L_i/C)²
```

Volume `V_i` grows roughly linearly in the order size and the aggressiveness of the quoting (tighter spreads → more fills → more fees), while `(2 − L_i/C)²` decays smoothly as `L_i` grows. Combined with the **top-5-days aggregation**, the search becomes: *for each of 5 "burst" days, find the parameterisation that pushes `L_i` to the point where the marginal volume gained per extra fee dollar equals the marginal `(1+ROE)²` decay*. The remaining ~55 days could be flat-ish — they don't enter the score. This is exactly the surface our Bayesian optimiser was tuning, and it is why every objective function in [`mm/example_optimizer.py`](mm/example_optimizer.py), [`mm/gtqb_xrp_optim.py`](mm/gtqb_xrp_optim.py) and [`demo.ipynb`](demo.ipynb) is some flavour of `Volume × (2 + Return)²` rather than the usual Sharpe-or-similar.

## Results

3rd place out of 50 teams, $5,000 prize. The pipeline is reproducible end-to-end on a single XRP/USD 2-hour slice — see **[demo.ipynb](demo.ipynb)**, which loads the data, runs the Avellaneda-Stoikov strategy, plots the dashboard, and runs an 8-trial Bayesian search using the competition's scoring rule as the objective.

**Order of magnitude.** A 15-hour XRP/USD backtest (2026-03-08, A-S with $500/quote order size, $2,000 inventory cap, full $3,000 book) gives:

| Metric | Value | Interpretation |
|---|---:|---|
| Daily Volume | **$112,917** | Volume traded (USD notional, daily-normalised) |
| Daily trade count | 306 | Filled orders |
| Net Return | **−1.90%** | $57 of the $3,000 burned (mostly maker fees) |
| Max Drawdown | 2.24% | $67 peak loss |
| `(1 + ROE)²` | 3.924 | Score multiplier (vs 4.0 at zero loss) |
| **Daily Score** | **$443,109** | `Volume × (1 + ROE)²` — the actual competition rule |
| Top-5-days projection | $2,215,543 | If five days held this regime |

These are demo-grade parameters, not the final competition configuration — but they show the regime the optimiser was hunting in: deliberately accepting a ~2% daily loss to multiply notional volume by an order of magnitude. The optimiser's job was to find the parameterisation that makes the marginal volume bought per extra fee dollar exactly offset the `(1+ROE)²` decay.

The same setup with conservative parameters (the 2h slice in [`demo.ipynb`](demo.ipynb)) gives Sharpe 25 / +0.11% return / $11k daily volume — a cleaner-looking strategy, but a much lower competition score, which is the point.

## What we'd do differently

- **Better σ estimation.** The current rolling-window sigma is naïve and noisy; a Hawkes-process or realised-variance estimator would track regime changes more cleanly and tighten spreads when it actually matters.
- **Richer alpha signals.** We only landed one (`OBISignal`) in the live bot. Trade flow toxicity (VPIN), book pressure asymmetry, and short-horizon momentum could each be plugged in via the existing [`AlphaSignal`](mm/mmbt/algorithms/signals/base_signal.py) interface — most of the plumbing is already there.
- **Dynamic risk controls.** Our position cap was static; in retrospect a drawdown-triggered kill switch and a position cap that scales with realised vol would have let us run hotter on calm days and pull back automatically when the regime shifted.
- **More CV.** We tuned on a handful of date ranges; an exhaustive walk-forward across the full Feb–Apr window — using the optimiser we already built — would have given us a much stronger signal about which params actually generalised.

## How to run

```bash
# 1. Create a virtual env (Python 3.13 required for the bundled qb/ C++ tracker)
python3.13 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Drop your own hftbacktest-format NPZ data into mm/data/{symbol}/
#    The competition data is not redistributable; see mm/example_data.py for the
#    live collector that produces the expected format.

# 4. Run the demo notebook
jupyter notebook demo.ipynb

# 5. Or run a single backtest script from inside mm/
cd mm && python as_xrp.py
```

To rebuild the qb/ C++ tracker from source (only needed if the bundled `.so` doesn't match your platform):

```bash
cd qb/src && bash tracker_pybind_mac.sh    # macOS
cd qb/src && bash tracker_pybind.sh        # Linux
```

## Acknowledgments

Team **Moon Shot** (Columbia MSFE):
- **Gunn Lee**
- **Maxime Geoffroy**
- **Hadrien Kremer**

The codebase was built collaboratively — strategies, optimiser, data pipeline, live bot, and C++ tracker were a joint effort across the three of us. Thanks to Gemini for hosting the competition and for the post-event feedback.

## Repository layout

```
gqtc-market_making/
├── demo.ipynb             ← Start here — end-to-end walkthrough
├── requirements.txt
├── mm/                    ← Research framework (mmbt) + scripts
│   ├── as_xrp.py          Avellaneda-Stoikov on XRP/USD (single asset)
│   ├── as_bnb.py          Avellaneda-Stoikov on BNB/USD
│   ├── gtqb_usdc_usdt.py  GTQB on stablecoin pair (multi-asset)
│   ├── gtqb_xrp_optim.py  Full Bayesian optim example (walk-forward CV)
│   ├── example_data.py    Live data collector (Gemini + Binance)
│   ├── example_optimizer.py
│   └── mmbt/              ← The package itself
│       ├── algorithms/    Strategies + pluggable alpha signals
│       ├── optimizer/     Bayesian search + cross-validation
│       ├── data/          WS connectors + NPZ writer
│       ├── runner.py      BacktestRunner + BacktestResult
│       ├── asset.py       AssetConfig
│       ├── metrics.py, charts.py, saver.py
│       └── …
└── qb/                    ← Live trading bot (production code)
    ├── quant_bot.py       Multi-coin async bot, mode dispatch
    ├── exchanges.py       Gemini REST + WS connector
    ├── dashboard.py       Real-time Dash analytics
    ├── tracker.*.so       Prebuilt C++ tracker (Linux + macOS, Python 3.13)
    └── src/               Tracker C++ source + build scripts
```
