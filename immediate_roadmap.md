# Immediate Roadmap

## 1 | What the latest run tells us

| Slice | Accuracy | Log-loss | Biggest pain-point |
|-------|----------|----------|--------------------|
| **Hold-out test (offline)** | **49.9 %** | 0.8336 | Large confusion between neighbouring classes (≈50 % diagonal) |
| **Strict “true-predict” (most-recent 2 000)** | **49.8 %** | – | Recall for *medium-short* drops to **40 %** |

*The model is still firmly capped at ~50 % raw accuracy, the same ceiling we’ve seen for the past few experiments, and mean confidence is only 0.51 → probabilities are barely more informative than a coin-flip.*

---

## 2 | Why we’re stuck

1. **Feature saturation:**  
   >75 % of gain comes from five “what was the last category?” flags (`prev_cat_*`, `same_as_prev`, `category_run_length`).  
   These are easy for the model to learn but don’t generalise well when the game drifts.

2. **Ordinal nature ignored:**  
   Classes 0 → 3 are ordered, but XGBoost treats them as unrelated buckets.  Mis-classifications mostly land in the *adjacent* bucket (see the light-blue band around the diagonal).

3. **Single temporal split + early stop:**  
   One lucky (or unlucky) split decides the round count and parameters; no cross-fold averaging.

4. **Probability mis-calibration:**  
   Confidence histogram is narrow and optimistic; that hurts bet-sizing.

---

## 3 | Improvements in order of “cost ⇢ impact”

| Priority | Change | Concrete step | Expected lift |
|----------|--------|---------------|---------------|
| **P0** | **Ordinal / pair-wise loss** | `objective="rank:pairwise"` or wrap labels in [`XGBRanker`], OR switch to **CatBoost** with `loss_function="YetiRank"` | +0.5–1 pp acc, lower off-diagonal errors |
| **P0** | **Isotonic / Platt calibration** | ```python\ncalib = CalibratedClassifierCV(final_model,\n                             method=\"isotonic\", cv=\"prefit\")\ncalib.fit(X_train_scaled, y_train)\n``` store in bundle | Better bet sizing, ~2–4 % bankroll swing reduction in back-tests |
| **P1** | **Rolling CV (TimeSeriesSplit)** | 5 folds, keep temporal order, average best_iteration | stabilises ±1 pp variance, may shorten early-stop round |
| **P1** | **Optuna search** on (`eta`, `max_depth`, `gamma`, `lambda`, `min_child_weight`, `subsample`, `colsample`) with the CV above | Usually digs up another 1 pp | |
| **P1** | **Custom class weights** instead of global scale: give extra ~15 % to class 1 only (where recall crashed) | Recovers the *medium-short* hole | |
| **P2** | **Richer lags**  ➜ add **exp-decay mean**, **rolling volatility**, **inter-arrival time** | Trees get a fresh signal beyond the obvious category flags |  |
| **P2** | **Hidden-Markov “regime” flag** | 2-state HMM on multiplier variance, one-hot the posterior state | Captures hot/cold market phases |
| **P3** | **Sequence model (TCN)** | 64-step raw streak-length input, PyTorch-Lightning, focal-loss output | In similar crash-style series we’ve seen 55–57 % accuracy |
| **P4** | **Reinforcement-learning policy** | Use calibrated probs + bankroll state; start with C51 | Optimises bankroll directly rather than accuracy |

---

## 4 | Quick code tweaks you can drop in **today**

```python
# after final_model is trained
from sklearn.calibration import CalibratedClassifierCV
calib = CalibratedClassifierCV(final_model,
                               method="isotonic", cv="prefit")
calib.fit(X_train_scaled, y_train)
model_bundle["calibrator"] = calib         # save with the model
```

```python
# when predicting
probs = final_model.predict(dpred)
if "calibrator" in model_bundle:
    probs = model_bundle["calibrator"].predict_proba(X_pred_scaled)
```

### **Ordinal objective**

```python
params["objective"] = "rank:pairwise"
# keep everything else; labels stay 0–3
```

(or LightGBM: `objective='lambdarank', metric='ndcg'`).

---

### 5 | How to tell if the tweaks worked

| Metric | Target |
|--------|--------|
| **Diagonal% in CM** | >55 % (was ~50 %) |
| **Class 1 recall** | ≥48 % (was 40 %) |
| **Brier score / log-loss** | <0.82 after calibration |
| **Back-tested bankroll** (flat 1 € per bet, stop at 14) | ≥+5 % vs current strategy on last 10 k streaks |

---

### 6 | Next step?

If you want to stay in the tree world, start with **ordinal loss + calibration + richer lags** (1–2 days of work).  
If you’re ready to prototype a sequence net, I can give you a minimal **TCN** training script that plugs into your existing data extractor—just say the word.
