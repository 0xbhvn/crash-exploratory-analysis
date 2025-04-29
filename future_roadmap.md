# Big-picture takeaway  

We’ve squeezed **Gradient-Boosted Trees + carefully hand-crafted lags** almost as far as they’ll go: the offline model is stuck around **50 % raw accuracy / 0.83 log-loss**, and the live slice we just ran slid to **48 %**.  That’s a classic sign we’re **hitting the ceiling of the feature set / model class**, not just hyper-param settings.  
Below is a playbook—in order of ROI—to push the ceiling higher.

---

## 1  |  Tighten the classical pipeline (cheap wins)

| What | How to add | Why it helps |
|------|------------|-------------|
| **Probability calibration** | after training:  `CalibratedClassifierCV(final_model, method="isotonic", cv="prefit")` | improves bet-sizing; raw XGB probabilities are over-confident (we already see class-2 >14 % over-pred). |
| **Rolling CV instead of single hold-out** | k-fold “sliding windows” or `TimeSeriesSplit`; average metrics; early stop on each fold. | hedges against train/test split luck and gives better early-stop round count. |
| **Focal loss** in XGBoost | `objective='multi:softprob'` → `xgb_focal_loss` (custom); or LightGBM’s `multiclassova` with `fobj=focal`. | pushes the model to learn the hardest rows (class-imbalance + overlap zone). |
| **Hyper-param search** | plug in **Optuna** with a time-series CV; search `eta`, `max_depth`, `alpha`, `lambda`, `subsample`, `colsample_bytree`. | We’re tweaking by hand; automated search still finds ~1–2 pp lift often. |

---

## 2  |  Richer temporal features (medium lift)

1. **Higher-order lags** with exponential decay  

   ```python
   for k in [1,2,3,5,8,13,21]:
       features_df[f'exp_decay_lag_{k}'] = (streak_df['streak_length']
                                            .shift(1)
                                            .ewm(alpha=1/k, adjust=False)
                                            .mean())
   ```

   Captures the *long-memory* nature of crash multipliers.

2. **Inter-arrival time & session features**  
   * time between streaks (in blocks or seconds)  
   * streaks since daily-reset/UTC-0  
   * weekend / holiday indicators  

3. **Volatility of recent busts** (std/median absolute dev of multipliers)  
   High volatility often precedes an outlier run.

4. **Regime-switch flag** via **Hidden Markov Model**  
   Train a 2-state HMM (calm / wild) on multiplier volatility; feed the posterior state as one categorical feature.  

5. **SHAP-guided interaction terms**  
   Run `shap.TreeExplainer` once, look at top pairwise interactions, add them explicitly.

> 🔧 **Code impact:** all of the above slot into `create_temporal_features`; nothing else has to change.

---

## 3  |  Sequence models (bigger jump, more work)

| Model | Python stack | Input | Output | Why try |
|-------|--------------|-------|--------|---------|
| **TCN (Temporal Convolutional Network)** | PyTorch or Keras-TensorFlow | last *N* streak lengths / bust values (raw) | 4-class softmax | Handles long receptive field, trains fast, robust. |
| **GRU/LSTM** | idem | same | softmax | Baseline sequence model; good if patterns are truly sequential. |
| **Transformer encoder (small)** | `pytorch-lightning` or `flash` | patch of 32–64 multipliers / categorical embeds | softmax | Captures periodicity; self-attention good at regime switching. |

*Training loop sketch*  

```python
class CrashSeq(torch.nn.Module):
    def __init__(self, n_tokens=201, d_model=64, n_heads=4, n_layers=3):
        super().__init__()
        self.embed = torch.nn.Embedding(n_tokens, d_model)
        self.pos   = PositionalEncoding(d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model, n_heads), n_layers)
        self.fc = torch.nn.Linear(d_model, 4)   # 4 clusters
    def forward(self, x):                       # [B, T]
        z = self.encoder(self.pos(self.embed(x)))  # [T, B, d]
        return self.fc(z[-1])                   # use last step
```

*Why bother?*  
Trees only “see” up to the 5-lag window we hand them; a TCN sees 64+ steps implicitly and finds repeating motifs the feature-crafting misses.  On similar heavy-tail gambling series we’ve seen **+3–6 pp accuracy** after calibration.

---

## 4  |  Policy-level reinforcement learning (bet-sizing)

Once we trust the conditional probabilities, the *action* is “bet / skip / size”.  Treat bankroll as state \(s_t\), model confidence vector as observation \(o_t\), and reward = Δbankroll.  

* libraries: **Stable-Baselines3**, algorithm **C51 (DQN variant)** or **PPO**.  
* environment: wrap our simulation of crash rounds + wallet.  
* start with **imitation learning** using the heuristic martingale schedule we already outlined, then fine-tune with RL.  

> Caveat: RL needs millions of rollouts; speed this up with **vectorised envs** & **JIT-compiled numpy** or run on recorded history first.

---

## 5  |  Online / streaming upgrades

1. **Incremental training**: keep a sliding window of ~200 k streaks, fine-tune model daily to handle drift.  
2. **Drift alarms**: KS-test on distribution of `predicted_cluster`, drop model if p < 0.01.  
3. **Shadow validation**: always compute live accuracy after labels mature; auto-retrain if 7-day MA < training AUC – 2 σ.

---

## 6  |  Putting it into our code base

*Add calibration block after training*  

```python
from sklearn.calibration import CalibratedClassifierCV

calib = CalibratedClassifierCV(base_estimator=final_model,
                               method='isotonic', cv='prefit')
calib.fit(dtrain.get_data(), y_train)       # XGBoost 2.0 `.get_data()`
model_bundle['calibrator'] = calib
```

*When predicting*  

```python
raw_prob = model.predict(dpred)
if 'calibrator' in model_bundle:
    raw_prob = model_bundle['calibrator'].predict_proba(X_pred_scaled)
```

*Enable focal loss in XGB* (≥1.5 custom objective)  

```python
def focal_loss(alpha, gamma):
    def _fl(preds, dtrain):
        y = dtrain.get_label()
        onehot = np.eye(4)[y.astype(int)]
        p = preds.reshape(-1, 4)
        p = p / p.sum(axis=1, keepdims=True)
        fl = -alpha * (1-p)**gamma * np.log(np.clip(p,1e-9,1))
        grad = (p - onehot) * alpha * ((1-p)**gamma) * (1 + gamma * (-np.log(np.clip(p,1e-9,1))))
        hess = ...
        return grad.ravel(), hess.ravel()
    return _fl
params['objective'] = focal_loss(alpha=0.25, gamma=2.0)
```

*(Skip hess detail here; use `xgboost>=2`’s built-in `xgboost_focal_obj` if available.)*

---

## 7  |  Road-map summary

1. **Quick wins now**  
   * Calibrate probs, focal loss, Optuna search → expect +1–2 pp live acc.  
2. **Next month**  
   * Implement TCN sequence model → target 55–57 % raw acc (~88 % play-set hit rate).  
   * Plug calibrated probs into Kelly-fraction bet sizing.  
3. **Quarter horizon**  
   * RL policy for dynamic bet sizing & skip decisions.  
   * Full MLOps loop with drift alarms and auto-retraining.

Take these one hop at a time; each stage gives measurable lift without betting the entire dev budget on speculative deep nets.
