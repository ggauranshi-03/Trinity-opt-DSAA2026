# Trinity — Revised Paper Plan (7 pages, IEEE two-column)

**Working title:** *Trinity: Zeroth-Order Curvature Sensing for Selective Natural-Gradient Preconditioning*

**Configuration (fixed):** ZO = RDSA second-difference **sensor**; SO = **K-FAC** (Fisher / information geometry); FO = **AdamW** actuator. Scope limited to existing assets: LT-CIFAR-10 (IF=100, 12,406 images), Wide-ResNet-101-2, ViT, ResNet-18.

---

## 0. Pre-work (do before writing a single line)

**P0. Reconcile the existing tables.** Current draft: ResNet-18 Adam = 43.88% (Table II) vs 55.77% (Table III); ViT Adam = 59.25% vs 41.74%. Same architecture, same data, same optimizer. Find the cause (LR mismatch, epoch budget, or evaluation bug). **If it is a bug, the ablation conclusions may also be invalid.** Nothing else should start until this is closed. Budget: 2–3 days.

**P1. Fix the measurement harness** (BN/dropout determinism — see §Pseudocode Notes N1–N3). A sensor that silently measures noise is the single highest-risk failure in this design.

**P2. Run A1 + B1 (see Ablation Plan).** These two decide whether the paper exists. If B1 fails, stop and reposition as a pure measurement paper (§Fallback).

---

## 1. Core contributions (state exactly these, in this order)

1. **A forward-only curvature sensor.** RDSA second differences yield unbiased estimates of `tr H` and `‖H‖_F²` per parameter block, hence a normalised effective rank `ρ ∈ (0,1]`, at ~6% overhead. Validated against exact Hessian-vector-product ground truth.
2. **A measurement study of Fisher anisotropy across blocks and training time.** Where in a network is the metric actually non-Euclidean? Answered empirically for three architectures under severe label imbalance. *This is the intellectual core and it stands alone.*
3. **Selective natural-gradient preconditioning (Trinity).** A per-block controller that spends a bounded K-FAC inversion budget where the sensor says the geometry is curved; AdamW elsewhere, with grafting so the switch is scale-safe. ZO additionally acts directly as a saddle-escape actuator.
4. **Cost result.** Match always-on K-FAC at matched wall-clock while inverting a fraction β_max of blocks.

**Framing sentence to reuse throughout:** *K-FAC approximates the Fisher; the ZO sensor measures Hessian anisotropy directly. Sensor and actuator speak the same geometric language — ρ→1 means the Fisher block is near-isotropic, so F⁻¹∇L ∝ ∇L and the inversion is wasted.* This alignment is what the Shampoo/SOAP route cannot claim, and it is the strongest single argument for the K-FAC choice.

**Claim NOT to make:** beating a well-tuned AdamW on accuracy. Report it if it appears; do not build the narrative on it.

---

## 2. Section plan with page budget

| § | Content | Pages |
|---|---|---|
| I | Introduction | 0.75 |
| II | Related work | 0.60 |
| III | Background & notation | 0.60 |
| IV | Method (incl. Algorithm 1 box) | 1.80 |
| V | Experimental setup | 0.45 |
| VI | Results | 1.20 |
| VII | Ablation study | 1.20 |
| VIII | Conclusion | 0.15 |
| — | References | 0.55 |
| | **Total** | **7.30** → trim in II and VI |

**Float budget: 6 maximum.** Fig.1 sensor validation · Fig.2 anisotropy heatmap · Fig.3 accuracy-vs-wall-clock Pareto · Table I main results · Table II gating ablation · Table III component ablation. Training-curve figures from the current draft (Figs 1–3) are cut — they consume a full column and say little.

---

### § I. Introduction (0.75 pp)

- **Para 1 — the tension.** First-order methods are geometry-blind; second-order preconditioning fixes that but pays O(n³) per block per inversion interval. The standard response is a global schedule (invert every τ steps, everywhere).
- **Para 2 — the observation.** Preconditioning value is *not uniform across blocks*. If a Fisher block is near-isotropic, F⁻¹∇L ∝ ∇L and the inversion buys nothing. Nobody measures this; the budget is spent uniformly.
- **Para 3 — why it isn't measured.** Anisotropy diagnostics normally need Hessian-vector products or eigendecompositions — as expensive as the preconditioning being decided about.
- **Para 4 — the insight.** Two forward passes give a *second* difference, not a first: an unbiased quadratic form `zᵀHz`. Same cost as a ZO gradient estimate; far more informative, because a two-point RDSA *gradient* estimator has O(d) variance and is dominated by backprop in a differentiable setting.
- **Para 5 — Trinity.** ZO senses, K-FAC and AdamW act, ZO also escapes saddles. All three orders present, each in its defensible role.
- **Para 6 — contributions** (the four above, bulleted).
- **Delete:** the duplicated roadmap paragraph. One roadmap sentence only, with numbering matching the actual sections.

### § II. Related work (0.60 pp)

Four compact paragraphs. Must cite:

- **FO / structural:** Adam, AdamW, Gradient Centralization.
- **SO:** Amari NGD, K-FAC, KFC (Grosse & Martens for conv), Osawa et al., **Shampoo (Gupta et al. 2018)**, **grafting (Agarwal et al.)**, **AdaHessian**, **SOAP (Vyas et al. 2024)**, **Muon**.
- **ZO / perturbation:** Spall (SPSA), Nesterov & Spokoiny, **MeZO (Malladi et al. 2023)**, **SAM (Foret et al. 2021)** ← mandatory, it owns the "escape sharp minima via an extra pass" motivation. **Ge et al. 2015 / Jin et al. 2017** for perturbed GD saddle escape.
- **Adaptive curvature use:** **Levenberg–Marquardt gain ratio** as the classical ancestor of curvature-driven adaptation, and K-FAC's own LM-style damping adaption. Position against it explicitly: our signal is *forward-only, dimension-normalised, and per-block*.

**Housekeeping:** remove Lookahead [9] and Socher et al. [10] (both uncited, the latter topically alien). Cite Kunstner et al. to **NeurIPS 2024**, not arXiv — and do **not** use it to explain SGD's collapse on a 10-class vision task; that paper concerns token-level imbalance in language models.

### § III. Background & notation (0.60 pp)

Reuse the current draft's Eqs. (1)–(3), (4)–(6), (8)–(13) — they are correct. Changes required:

- **Fix Eq. (7) (GC).** Currently malformed: free index `i` appears on the RHS but not the LHS, and a scalar mean is subtracted from a tensor. Write it per output unit, and make it consistent with Eq. (24)'s axis convention (currently the two disagree).
- **Add:** block partition θ = {θ^(ℓ)}, sizes d_ℓ, eligibility (ndim ≥ 2 for SO; 1-D blocks permanently FO).
- **Cut:** Section III-D (RDSA gradient estimator, Eqs. 14–16). It is no longer used as an estimator. Replace with the second-difference identity, which belongs in §IV-B anyway. Saves ~0.2 pp.

### § IV. Method (1.80 pp)

**IV-A. Design principle (0.2 pp).** Role reassignment: ZO senses, FO/SO act. One sentence on why the additive tri-order composition is the wrong construction (O(d) estimator variance; layer-wise rescaling of a globally-computed directional derivative biases the descent direction).

**IV-B. ZO curvature sensor (0.5 pp).** Second-difference identity, the two moment estimators, the debiased `ρ̂`, the negative-curvature fraction `ν̂`. **Include the three correctness requirements as a numbered remark** — shared minibatch, fp32/fp64 accumulation, ε ≈ 10⁻². These are non-obvious, they are silent failure modes, and stating them is itself a contribution to reproducibility.

**IV-C. Information-geometric interpretation (0.25 pp).** ρ→1 ⇒ Fisher block ≈ cI ⇒ natural gradient degenerates to Euclidean ⇒ inversion wasted. ρ≪1 ⇒ metric strongly curved ⇒ Kronecker structure has something to exploit. **This is the paragraph that justifies K-FAC over Shampoo/SOAP** — the sensor measures the same object the actuator approximates.

**IV-D. Round-robin scheduling (0.15 pp).** Per-block probes cannot share forward passes (a joint perturbation measures cross-block terms, not diagonal blocks). Sensing all B blocks per event costs (2m+1)·B forwards; round-robin over n_p blocks keeps overhead at ~6%. Non-obvious — state it.

**IV-E. Controller (0.25 pp).** Hysteresis (ρ_lo < ρ_hi), dwell time τ_dwell, global budget β_max with smallest-ρ admission. β_max is the dial the cost claim is measured against.

**IV-F. Grafting (0.2 pp).** FO and SO modes live on different scales; a raw switch jumps the effective step size by an unknown layer-dependent factor. Direction from active mode, magnitude from a shadow AdamW state maintained for every block.

**IV-G. ZO as escape actuator (0.15 pp).** Trigger on ν̃ ≥ ν_thr AND small gradient norm. Cite Jin et al. 2017. Note the lockout counter.

**IV-H. Algorithm 1 box (0.4 pp).**

**Reframe, do not delete, `F_collect`.** In the current draft it is sold as a headline contribution ("non-interfering multi-order protocol"). Demote it to a one-sentence implementation note in IV-B with a defensible purpose: it protects the Fisher statistics from the sensor's off-path forward passes.

### § V. Experimental setup (0.45 pp)

- Dataset: LT-CIFAR-10, IF=100, 12,406 images. Standard crop + flip. **Say "data imbalance factor", not "architectural".**
- Architectures: ResNet-18 (11M), Wide-ResNet-101-2 (127M, 3×3 stem), ViT (86M, patch 4, 12 layers, dim 768, pre-LN).
- **Hardware — currently absent from the draft and mandatory.** GPU model, count, PyTorch/CUDA versions.
- **3 seeds minimum**, mean ± std on every reported number.
- **Matched Optuna trial budgets across all optimizers**, with the trial count stated. The current asymmetry (2 params for baselines vs 5 for Trinity) is a confound reviewers will name.
- Baselines: AdamW, SAM, **K-FAC always-on** (currently missing from the headline tables despite beating full Trinity in your own ablation), grafted Shampoo if time permits.
- **Matched wall-clock** for headline comparisons, not matched epochs.
- Report all Trinity hyperparameters including **τ_Z and ε_Z**, which the current draft never gives values for.

### § VI. Results (1.20 pp)

**VI-A. Sensor validation (Fig. 1, 0.35 pp).** ZO ρ̂ vs HVP ground truth, per block, at epochs {1,10,50,100}, three architectures. Spearman ρ + scatter. *Lead with this.* If the correlation is strong the paper has a result regardless of what the optimizer does.

**VI-B. Anisotropy measurement study (Fig. 2, 0.4 pp).** Heatmap: blocks × training time, grouped by layer type (stem/patch-embed, QKV, MLP, norm, head). Expected finding: anisotropy spans orders of magnitude across blocks and drifts over training. This *justifies* per-block gating on evidence instead of assertion, and is the paper's most quotable figure.

**VI-C. Main comparison (Table I, 0.3 pp).** 3 architectures × {AdamW, SAM, K-FAC always-on, Trinity}, accuracy ± std at matched wall-clock, plus inversion count and peak memory.

**VI-D. Cost frontier (Fig. 3, 0.15 pp).** Accuracy vs wall-clock as β_max sweeps 0 → 1. β_max=0 is AdamW, β_max=1 is always-on K-FAC. **This is the headline figure for contribution 4.**

### § VII. Ablation study (1.20 pp)

Tables II and III (below) plus ~0.5 pp of discussion. **Rule: do not interpret any gap smaller than the reported standard deviation.** The current draft interprets a 0.06-point difference as evidence of saddle escape; that must not recur.

### § VIII. Conclusion (0.15 pp)

Three sentences plus one signposting the journal extension: uniform per-mode sufficient decrease + minimum dwell time ⇒ segment-wise O(1/√T) nonconvex rate; escape mode upgrades to second-order stationarity. Do not attempt it here.

---

## 3. Pseudocode

Algorithm 1 goes in the paper. Algorithms 2–3 go in the paper if space allows, otherwise a footnote pointing to the repo. All are written to be transcribed directly into PyTorch.

```
════════════════════════════════════════════════════════════════════════════
ALGORITHM 1 — Trinity
════════════════════════════════════════════════════════════════════════════
Require: θ⁽⁰⁾; LR schedule η⁽ᵗ⁾; loss L(·); blocks {ℓ}, sizes d_ℓ
Require: AdamW  β₁, β₂, ε_a, λ_wd
Require: Sensor τ_Z, m, ε_Z, n_p
Require: Control ρ_lo, ρ_hi, α_ema, τ_dwell, β_max
Require: K-FAC  τ_stat, τ_inv, ρ_kf, λ_damp, d_max
Require: Escape ν_thr, γ_g, r_esc, τ_esc

 1: for all ℓ:  M, V ← 0                      ▷ in-basis Adam moments (SO branch)
 2: for all ℓ:  M_ad, V_ad ← 0                ▷ shadow AdamW  (grafting magnitude)
 3: for all ℓ:  Â, Ŝ ← I ;  A_inv, S_inv ← I  ▷ Kronecker factors + inverses
 4: for all ℓ:  mode ← FO ; ρ̃ ← 1 ; ν̃ ← 0 ; dwell ← 0 ; lock ← 0 ; ḡ ← 0
 5: F_collect ← True ;  t ← 0
 6: while not converged do
 7:     t ← t + 1 ;  sample minibatch B⁽ᵗ⁾
 8:     L_base ← L(θ⁽ᵗ⁾; B⁽ᵗ⁾) ;  g ← ∇_θ L_base       ▷ hooks cache A_raw, S_raw
 9:     if use_GC then  g⁽ℓ⁾ ← CENTRALIZE(g⁽ℓ⁾)  ∀ℓ, ndim ≥ 2
                                                 ▷ on the RAW gradient (Yong et al.)
   ┌──────────────── PHASE 1 · ZO sensing  (every τ_Z steps) ────────────────┐
10:     if t mod τ_Z = 0 then
11:         F_collect ← False                    ▷ freeze Fisher statistics
12:         S_t ← next n_p eligible blocks (round-robin)
13:         for ℓ ∈ S_t do
14:             (ρ̂, ν̂) ← ZO-PROBE(θ⁽ᵗ⁾, B⁽ᵗ⁾, ℓ, m, ε_Z)          ▷ Alg. 2
15:             ρ̃⁽ℓ⁾ ← α_ema·ρ̃⁽ℓ⁾ + (1−α_ema)·ρ̂ ;   ν̃⁽ℓ⁾ ← ν̂
16:         end for
17:         F_collect ← True                     ▷ cache untouched by probes
18:         {mode⁽ℓ⁾} ← CONTROLLER({ρ̃}, {dwell}, β_max)             ▷ Alg. 3
19:     end if
   ├──────────────── PHASE 2 · ZO actuator (saddle escape) ─────────────────┤
20:     for ℓ ∈ S_t sensed at this step do
21:         if ν̃⁽ℓ⁾ ≥ ν_thr  and  ‖g⁽ℓ⁾‖₂ ≤ γ_g·ḡ⁽ℓ⁾  and  lock⁽ℓ⁾ = 0 then
22:             u ~ Uniform(S^{d_ℓ−1})
23:             θ⁽ℓ⁾ ← θ⁽ℓ⁾ + r_esc·‖θ⁽ℓ⁾‖₂·u       ▷ perturbed GD, Jin et al. 2017
24:             M⁽ℓ⁾ ← ½M⁽ℓ⁾ ;  M_ad⁽ℓ⁾ ← ½M_ad⁽ℓ⁾  ▷ damp now-stale momentum
25:             lock⁽ℓ⁾ ← τ_esc
26:         end if
27:     end for
28:     for all ℓ:  lock⁽ℓ⁾ ← max(lock⁽ℓ⁾ − 1, 0) ;  ḡ⁽ℓ⁾ ← EMA(‖g⁽ℓ⁾‖₂)
   ├──────────────── PHASE 3 · Fisher factors and inverses ─────────────────┤
29:     if t mod τ_stat = 0 then
30:         for all eligible ℓ do                        ▷ cheap: outer products
31:             Â⁽ℓ⁾ ← ρ_kf·Â⁽ℓ⁾ + (1−ρ_kf)·A_rawᵀA_raw
32:             Ŝ⁽ℓ⁾ ← ρ_kf·Ŝ⁽ℓ⁾ + (1−ρ_kf)·S_rawᵀS_raw
33:         end for
34:     end if
35:     if t mod τ_inv = 0 then
36:         for ℓ with mode⁽ℓ⁾ = SO and dim(ℓ) ≤ d_max do
37:             π ← sqrt( tr(Â)·dim(Ŝ) / (tr(Ŝ)·dim(Â) + δ) )     ▷ trace balance
38:             A_inv⁽ℓ⁾ ← (Â + √λ_damp·π·I)⁻¹        ▷ ◄ GATED: the saved O(n³)
39:             S_inv⁽ℓ⁾ ← (Ŝ + (√λ_damp/π)·I)⁻¹
40:         end for
41:     end if
   ├──────────────── PHASE 4 · update engine with grafting ─────────────────┤
42:     for all ℓ do
43:         ▷ shadow AdamW branch — always maintained, supplies the magnitude
44:         M_ad ← β₁M_ad + (1−β₁)·g⁽ℓ⁾
45:         V_ad ← β₂V_ad + (1−β₂)·(g⁽ℓ⁾ ⊙ g⁽ℓ⁾)
46:         Δ_ad ← (M_ad /(1−β₁ᵗ)) ⊘ ( sqrt(V_ad /(1−β₂ᵗ)) + ε_a )
47:         if mode⁽ℓ⁾ = SO then
48:             C ← [ G⁽ℓ⁾  g_b⁽ℓ⁾ ]                       ▷ weight ‖ bias
49:             G_pre, g_b,pre ← Split( S_inv⁽ℓ⁾ · C · A_inv⁽ℓ⁾ )
50:             M ← β₁M + (1−β₁)·G_pre
51:             V ← β₂V + (1−β₂)·(G_pre ⊙ G_pre)
52:             Δ_act ← (M /(1−β₁ᵗ)) ⊘ ( sqrt(V /(1−β₂ᵗ)) + ε_a )
53:         else
54:             Δ_act ← Δ_ad ;   M ← M_ad ;  V ← V_ad     ▷ keep SO state warm
55:         end if
56:         ▷ GRAFT: direction from actuator, magnitude from AdamW
57:         Δ ← ‖Δ_ad‖_F · Δ_act / (‖Δ_act‖_F + δ)
58:         θ⁽ℓ⁾ ← θ⁽ℓ⁾ − η⁽ᵗ⁾·( Δ + λ_wd·θ⁽ℓ⁾ )          ▷ decoupled weight decay
59:     end for
60: end while
════════════════════════════════════════════════════════════════════════════
```

```
════════════════════════════════════════════════════════════════════════════
ALGORITHM 2 — ZO-PROBE   (forward-only curvature sensor, one block)
════════════════════════════════════════════════════════════════════════════
Require: θ, minibatch B, block ℓ, probes m, radius ε_Z
Assumes: F_collect = False ; dropout in eval ; BN running-stat updates frozen

 1: L₀ ← L(θ; B)                       ▷ recompute under the probe configuration
 2: for i = 1 … m do
 3:     z_i ~ N(0, I_{d_ℓ})            ▷ supported on block ℓ ONLY
 4:     L⁺ ← L(θ + ε_Z·z_i ; B)        ▷ SAME B for every probe
 5:     L⁻ ← L(θ − ε_Z·z_i ; B)        ▷ accumulate loss in fp32/fp64
 6:     c_i ← (L⁺ − 2·L₀ + L⁻) / ε_Z²  ▷ ≈ z_iᵀ H⁽ℓ⁾ z_i
 7: end for
 8: c̄ ← mean(c) ;  s² ← var(c, ddof = 1)
 9: r̂_eff ← 2·max(c̄² − s²/m, 0) / (s² + δ)     ▷ debiased (tr H)² / ‖H‖_F²
10: ρ̂ ← clip( r̂_eff / d_ℓ , 1/d_ℓ , 1 )
11: ν̂ ← |{ i : c_i < 0 }| / m
12: return (ρ̂, ν̂)
════════════════════════════════════════════════════════════════════════════
Cost: 2m + 1 forward passes per probed block, no backward pass.
Identities:  E[c] = tr H⁽ℓ⁾ ,  Var[c] = 2‖H⁽ℓ⁾‖_F²  for z ~ N(0, I).
The −s²/m term removes the upward bias of c̄² as an estimator of (tr H)².
════════════════════════════════════════════════════════════════════════════
```

```
════════════════════════════════════════════════════════════════════════════
ALGORITHM 3 — CONTROLLER   (hysteresis · dwell time · inversion budget)
════════════════════════════════════════════════════════════════════════════
Require: {ρ̃⁽ℓ⁾}, {mode⁽ℓ⁾}, {dwell⁽ℓ⁾}, ρ_lo < ρ_hi, τ_dwell, β_max, B blocks

 1: cand ← ∅
 2: for all eligible ℓ do
 3:     dwell⁽ℓ⁾ ← dwell⁽ℓ⁾ + 1
 4:     if dwell⁽ℓ⁾ < τ_dwell then
 5:         if mode⁽ℓ⁾ = SO then cand ← cand ∪ {ℓ}          ▷ locked in
 6:         continue
 7:     end if
 8:     if mode⁽ℓ⁾ = FO and ρ̃⁽ℓ⁾ <  ρ_lo then cand ← cand ∪ {ℓ}   ▷ turn ON
 9:     if mode⁽ℓ⁾ = SO and ρ̃⁽ℓ⁾ ≤  ρ_hi then cand ← cand ∪ {ℓ}   ▷ stay ON
10: end for                                ▷ ρ_lo < ρ_hi ⇒ no chattering
11: if |cand| > ⌊β_max·B⌋ then
12:     cand ← the ⌊β_max·B⌋ members of cand with smallest ρ̃      ▷ budget
13: end if
14: for all eligible ℓ do
15:     new ← SO if ℓ ∈ cand else FO
16:     if new ≠ mode⁽ℓ⁾ then  dwell⁽ℓ⁾ ← 0 ;  mode⁽ℓ⁾ ← new
17: end for
18: return {mode⁽ℓ⁾}
════════════════════════════════════════════════════════════════════════════
```

### Implementation notes (PyTorch) — read before coding

**N1 · BatchNorm.** Probe forwards would update BN running statistics and corrupt training. Set `bn.momentum = 0` (or cache and restore `running_mean/var`) for the probe window. Do **not** simply call `model.eval()`, as that switches BN to running stats and changes the function being probed relative to the training forward.

**N2 · Dropout.** Stochastic masks will dominate the second difference. Put dropout modules in `eval()` for the probe window. Because the configuration differs from the training forward, **recompute `L₀` inside the probe** (Alg. 2 line 1) rather than reusing `L_base`. This costs one extra forward per probed block and is what makes the estimate valid.

**N3 · Numerics.** `ε_Z ≈ 1e-2`. The second difference divides by `ε_Z²`, so relative error grows as `ε_mach/ε_Z²`; the usable fp32 window is `ε ~ ε_mach^(1/4)`. Accumulate probe losses in fp32/fp64 even under bf16 training. Tuning `ε_Z` *down* "for accuracy" produces pure noise that looks exactly like a working sensor.

**N4 · Perturbation memory.** Do not materialise `z` for a 127M-parameter model. Seed a `torch.Generator`, regenerate `z` on the fly for the `+`, `−`, and restore passes. Restore by re-adding rather than by keeping a full parameter copy.

**N5 · Hooks.** Forward hooks still fire under `torch.no_grad()`; backward hooks do not. `F_collect` must gate the **forward** hook at minimum. Guard both for safety.

**N6 · Block definition.** One block per weight tensor. Bias handled jointly with its weight via the `[G  g_b]` concatenation (Alg. 1 line 48), as in standard K-FAC. 1-D blocks (norm gains, standalone biases) are permanently FO.

**N7 · Grafting norms.** Frobenius norm per block, not global. `δ = 1e-12`.

**N8 · State when FO.** Lines 54 keeps `M, V` synced to the shadow AdamW state so that a later FO→SO switch starts warm rather than from zero. Ablate against a cold reset (D7).

**N9 · Conv layers.** K-FAC for Conv2d requires the KFC treatment (Grosse & Martens 2016) — patch extraction before forming Â. Reuse existing code; do not silently fall back to identity, or the WRN result measures AdamW.

**N10 · `d_max` fallback.** Blocks exceeding `d_max` are permanently FO. **Log the fraction of parameters this excludes and report it** — the current draft's silent fallback means much of the ViT may never have been preconditioned.

### Default hyperparameters (report all of these)

| Group | Values |
|---|---|
| Sensor | `m=4`, `n_p=2`, `τ_Z=100`, `ε_Z=1e-2` |
| Controller | `α_ema=0.7`, `ρ_lo=0.15`, `ρ_hi=0.35`, `τ_dwell=500`, `β_max=0.3` |
| K-FAC | `τ_stat=10`, `τ_inv=100`, `ρ_kf=0.95`, `λ_damp=1e-2`, `d_max` per arch |
| Escape | `ν_thr=0.75`, `γ_g=0.3`, `r_esc=1e-3`, `τ_esc=200` |
| AdamW | `β₁=0.9`, `β₂=0.999`, `ε_a=1e-8`, `λ_wd` tuned |

**Sensor overhead:** `(2m+1)·n_p / (3·τ_Z) = 18/300 ≈ 6%`. Put this number in the abstract.

---

## 4. Ablation plan

Sized to ~1.2 pages. **Groups A–D go in the paper; E and F are repo-only** unless space frees up.

### Priority order under compute constraint
`A1 → B1 → A5 → C1 → D-table`. Those five carry the paper.

### Group A — Is the sensor real? *(Fig. 1, Fig. 2)*

| # | Experiment | Purpose |
|---|---|---|
| **A1** ★ | ZO `ρ̂` vs HVP ground truth, per block, epochs {1,10,50,100}, 3 archs. Spearman + scatter. | **Paper-saving.** Establishes the instrument. Stands alone even if the optimizer shows no gain. |
| **A2** | `m ∈ {2,4,8,16}` — correlation vs cost | Justifies `m=4` |
| **A3** | `ε_Z ∈ {1e-1,1e-2,1e-3,1e-4}` × {fp32, bf16} accumulation | Expect a U-shape. **Genuinely novel** — nobody documents the usable window for second-difference probes at scale |
| **A4** | Shared vs independent minibatch across the `m` probes | Demonstrates the variance-contamination failure mode |
| **A5** ★ | Anisotropy heatmap: blocks × training time, grouped by layer type | **The most quotable figure in the paper** |

### Group B — Is the gating signal doing work? *(Table II)*

| # | Variant | Purpose |
|---|---|---|
| **B1** ★★ | **Random gating at matched β_max** | **Decides the paper.** If Trinity does not beat this, the finding is "less preconditioning is fine", not "the sensor works". **Run this second, before investing further.** |
| **B2** | Inverted gating (largest ρ̃) | Should be clearly worse — confirms directionality, not mere selectivity |
| **B3** | Oracle gating (HVP-derived ρ) | Upper bound. If oracle ≫ ZO, the sensor is the bottleneck — report that honestly |
| **B4** | Static heuristic (β_max largest blocks by parameter count) | The cheap baseline a reviewer will propose |
| **B5** | Gradient-covariance gating (off-diagonal mass of Â, Ŝ — free) | **Be prepared for this to be competitive.** If it is, reposition ZO's contribution around negative-curvature detection, which `E[ggᵀ] ⪰ 0` provably cannot see |

### Group C — Budget and controller *(Fig. 3 + text)*

| # | Sweep | Purpose |
|---|---|---|
| **C1** ★ | `β_max ∈ {0, 0.1, 0.2, 0.3, 0.5, 1.0}`, accuracy vs wall-clock | **Headline figure.** `0` = AdamW, `1` = always-on K-FAC |
| **C2** | `ρ_lo = ρ_hi` vs separated; log switch counts | Chattering |
| **C3** | `τ_dwell ∈ {0, 100, 500, 2000}` | — |
| **C4** | `τ_Z ∈ {50, 100, 500, 2000}` | Overhead vs staleness |
| **C5** | `n_p ∈ {1, 2, 4, all}` | Round-robin width |

*Report C2–C5 as a compact 4-row table or in text; do not spend a float on them.*

### Group D — Component isolation *(Table III — the tri-order claim)*

| Variant | ZO sense | ZO escape | K-FAC | GC | Graft | Tests |
|---|---|---|---|---|---|---|
| AdamW | – | – | – | – | – | floor |
| K-FAC always-on | – | – | all | ✓ | ✓ | ceiling / cost reference |
| **Trinity (full)** | ✓ | ✓ | gated | ✓ | ✓ | proposed |
| − escape | ✓ | – | gated | ✓ | ✓ | ZO-as-actuator |
| − GC | ✓ | ✓ | gated | – | ✓ | **fills the 4th-component gap in the current draft** |
| − grafting | ✓ | ✓ | gated | ✓ | – | switch discontinuity |
| escape only | ✓ | ✓ | – | ✓ | – | ZO alone over AdamW |

**D6.** Escape diagnostics: trigger counts, and loss / ν̃ traces around triggered escapes. **If escapes fire rarely, say so and scope the claim** rather than over-reading a sub-noise difference — this is exactly where the current draft went wrong (0.06 pp interpreted as saddle escape).

**D7.** Grafting alternatives: graft vs no-graft vs reset moments at switch.

### Group E / F — repo appendix only (scope-limited paper)

**E:** cross-architecture gating patterns; imbalance factor `{1, 10, 100}`; batch size `{128, 512, 2048}` (strongest documented regime for preconditioning — if gains grow with batch size, that is memorable, and it costs no new dataset).

**F:** wall-clock breakdown (fwd/bwd · sensing · statistics · inversions · update); peak memory vs always-on K-FAC; realised vs budgeted inversion count.

---

## 5. Non-negotiable protocol

- 3 seeds minimum; mean ± std everywhere; **no interpretation of gaps below the std**.
- Matched Optuna trial budgets, count stated.
- Matched wall-clock for headline comparisons; hardware reported.
- **K-FAC always-on must appear as a baseline in the main table**, not only in the ablation.
- Every hyperparameter given a value, including `τ_Z` and `ε_Z`.
- Wall-clock numbers must be internally consistent across architectures (the current draft has ResNet-18 at ~8× the per-epoch cost of Wide-ResNet-101-2).

---

## 6. Scope note and one cheap mitigation

You have chosen to keep LT-CIFAR-10 + the three existing architectures. That is defensible **because the paper's primary claim is a measurement claim**, and a measurement study on three architectures at one imbalance level is a coherent unit of work. Frame it that way in the limitations sentence: *"we characterise Fisher anisotropy for three architectures under severe label imbalance; whether the observed block-level patterns transfer across data regimes is left open."*

The one addition I would still argue for, because it costs almost nothing: **CIFAR-100-LT reuses your exact pipeline** with a change to the class count and a re-partition. It converts "one dataset" into "two datasets", removes the single most common desk-reject trigger for empirical optimizer papers, and — because C=100 versus C=10 changes the head block's geometry substantially — it likely strengthens A5. Roughly one extra day of compute per architecture. Your call, but it is a high return for the cost.

---

## 7. Expected outcome and fallback

**Most likely result:** C1 shows Trinity matching always-on K-FAC at a fraction of the inversions; accuracy against a well-tuned AdamW is a wash. Plan the narrative for that: lead with A5, headline C1, treat any accuracy win as incidental.

**If B1 fails** (Trinity ≈ random gating): do not force the optimizer claim. Reposition as a pure measurement paper — *"Where is the Fisher actually curved? A cheap forward-only survey"* — built on A1, A3, A5 plus the negative result from B1. That is still publishable, still honest, and the negative result has real value for anyone building adaptive-preconditioning schedules.

**Do not claim** to beat AdamW on accuracy. That claim gets judged against grafted Shampoo and SOAP, and this evidence base cannot win that fight.
