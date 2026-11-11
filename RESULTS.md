# Practical 6 — results log (xmzj54)

## Environment (recorded 2026-11-10, turn 001)

| Item | Value |
|---|---|
| Host | gpu10 (NCC, SLURM job, 24 h limit) |
| GPU | NVIDIA A100 80GB PCIe, **MIG slice 1g.10gb**: 14 SMs (1/7 of the A100), 9.7 GiB memory |
| Driver / CUDA | 610.43.02 / CUDA 13.3 UMD; torch built for CUDA 13.0 |
| CPU | 4 cores allocated (SLURM cpu=4, `nproc` = 4), 20 GB RAM allocated |
| Python | 3.12.3 (`~/deep_learning` venv) |
| torch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 |
| torch-fidelity | 0.4.0 (Inception weights cached) |
| Note | The MIG slice is shared: processes of other users (zrgj62, cljb35) also show on it, so speeds can vary |

## All runs

KID x 1000 on 1,000 samples, validation split unless stated. Minutes = runner wall clock (train + sample + KID).

| exp | task | params | updates | seed | split | KID x1000 | minutes | notes |
|---|---|---|---|---|---|---|---|---|
| exp_000_ddpm | ddpm | 120,075 | 4,000 | 3801 | val | 164.3 | 4.2 | starter as-is (finished before turn 001 was stopped; sheet says 220.0) |
| exp_001–004 | — | — | — | — | — | — | — | killed at the start of turn 002 (plan changed; not needed) |
| exp_010_ddpm_strong | ddpm | 981,651 (+16 buf) | 20,000 | 3801 | val | **100.9** (EMA .9998) / 105.0 (.999) / 104.5 (.9995) / 188.6 (.9999) | 29.0 (+3.1 extra EMAs) | strong colour casts (saturated blue/magenta/green tints) |
| exp_011_flow_strong | flow | 981,651 (+16 buf) | 20,000 | 3801 | val | 25.9 (EMA .9998) / 14.7 (.999) / **14.4** (.9995) / 355.3 (.9999) | 28.0 (+1.9 extra EMAs) | .9995: clean, varied horses, natural colours; .9999: noise-textured mush |
| exp_012_flow_cosine | flow | 981,651 (+16 buf) | 20,000 | 3801 | val | 18.2 (EMA .999) / **18.0** (.9995) / 18.8 (.9998) / 27.5 (.9999) | 26.9 (+1.9 extra EMAs) | lr 2e-3 + cosine to 0, warm-started EMA; clean, similar to 011 |
| exp_013_flow_wide | flow | 950,363 (+24 buf) | 20,000 | 3801 | val | 20.9 (EMA .999) / **20.8** (.9995) / 26.2 (.9998) / 53.9 (.9999) | 19.7 (+2.0 extra EMAs) | 012 recipe, (48,56,56), 1 down / 2 up, attn 8x8; clean, similar |
| exp_014_flow_xpred | flow | 981,651 (+16 buf) | 20,000 | 3801 | val | 30.0 (EMA .999) / **29.7** (.9995) / 49.5 (.9998) / 131.3 (.9999) | 27.5 (+1.0 extra EMAs) | 011 net + recipe, x-prediction → velocity in `forward`; softer, blotchy textures |
| exp_015_flow_mp | flow | 927,218 (+16 buf) | 20,000 | 3801 | val | 16.2 (EMA .999) / **14.8** (.9995) / 16.0 (.9998) / 21.5 (.9999) | 25.9 (+1.5 extra EMAs) | EDM2 MP U-Net, lr 1e-2 constant; clean, varied, similar to 011 |
| exp_016_flow_lrtail | flow | 981,651 (+16 buf) | 20,000 | 3801 | val | 17.1 (EMA .999) / **15.7** (.9995) / 19.2 (.9998) / 50.1 (.9999) | 30.4 (+1.0 extra EMAs) | 011 + lr 1e-3 constant then linear to 0 over 16k-20k; clean, like 011 |
| exp_017_flow_mp_isqrt | flow | 927,218 (+16 buf) | 20,000 | 3801 | val | 25.3 (EMA .999) / 24.3 (.9995) / **23.7** (.9998) / 30.9 (.9999) | 28.5 (+1.7 extra EMAs) | 015 + EDM2 1/sqrt(t) decay (T_REF 1500, lr ends 2.7e-3); grainy, dithered textures |
| exp_018_flow_muon | flow | 981,651 (+16 buf) | 20,000 | 3801 | val | **34.2** (EMA .999) / 37.6 (.9995) / 80.0 (.9998) / 223.2 (.9999) | 31.5 (+1.8 extra EMAs) | 011 + MuonAdamW (Muon lr 0.02); muddy, desaturated, noisy textures, broken shapes |
| exp_019_flow_dropout | flow | 981,651 (+16 buf) | 20,000 | 3801 | val | **14.4** (EMA .999) / 14.5 (.9995) / 20.1 (.9998) / 45.5 (.9999) | 31.7 (+1.7 extra EMAs) | 011 + dropout 0.1 before conv2; clean, varied, slightly smoother than 011 |
| exp_021_flow_hires | flow | 958,179 (+16 buf) | 20,000 | 3801 | val | **15.6** (EMA .999) / 16.0 (.9995) / 21.7 (.9998) / 35.7 (.9999) | 33.3 (+1.0 extra EMAs) | widths (40, 48, 40); crisper fine detail, a few more broken shapes. Rerun after the runner bug (first attempt stopped at 14k) |
| exp_023_flow_noflip | flow | 981,651 (+16 buf) | 20,000 | 3801 | val | 15.8 (EMA .999) / **15.3** (.9995) / 17.2 (.9998) / 47.6 (.9999) | 28.3 (+2.0 extra EMAs) | 011 without flips; clean, like 011, slightly softer. Rerun after the runner bug (crashed attempt: .9995 = 15.27) |
| exp_024_flow_beta2 | flow | 981,651 (+16 buf) | 20,000 | 3801 | val | 15.2 (.999) / 14.6 (.9993) / **14.2** (.9995) / 14.7 (.9997) / 16.8 (.9998) | 26.3 (+1.9 extra EMAs) | 011 + AdamW beta2 0.999; clean, like 011 |
| exp_026_flow_lowres | flow | 944,755 (+16 buf) | 20,000 | 3801 | val | 19.7 (.999) / 19.4 (.9993) / **19.3** (.9995) / 19.8 (.9997) / 22.5 (.9998) | 25.3 (+2.2 extra EMAs) | (32, 40, 40, 40), blocks (1, 2, 2, 2), attn 16/8/4; softer, blurrier detail, shapes no better |
| exp_027_flow_convnext | flow | 935,691 (+24 buf) | 20,000 | 3801 | val | **16.0** (.999) / 16.1 (.9993) / 16.3 (.9995) / 20.3 (.9997) | 63.4 (+2.7 extra EMAs) | ConvNeXt blocks, (48, 56, 64), 2/level, expand 4; clean, slightly soft. ~5.4 upd/s shared |
| exp_029_flow_convnext_deep | flow | 895,491 (+20 buf) | 20,000 | 3801 | val | **16.3** (.999) / 16.7 (.9993) / 17.4 (.9995) / 22.2 (.9997) | 69.2 (+1.3 extra EMAs) | ConvNeXt blocks, (40, 56, 56), 3/level, expand 3; clean, smooth, a little blurrier. ~4.6 upd/s shared |
| exp_030_flow_precond | flow | 981,651 (+16 buf) | 20,000 | 3801 | val | 19.9 (.999) / **19.5** (.9993) / 19.6 (.9995) / 21.2 (.9997); snapshots .9995: 28.5 @10k, 21.2 @15k | 27.0 (+3.3 extra EMAs/snapshots) | 011 + EDM-style preconditioning; softer, partly washed out / desaturated |
| exp_031_flow_attn32 | flow | 985,939 (+16 buf) | 20,000 | 3801 | val | 16.9 (.999) / 16.5 (.9993) / **16.1** (.9995) / 16.9 (.9997); snapshots .9995: 24.3 @10k, 17.6 @15k | 33.5 (+1.9 extra EMAs/snapshots) | 011 + attention in the last 32x32 encoder block; clean, sharp, like 011 |
| exp_033_flow_lean | flow | 984,711 (+20 buf) | 20,000 | 3801 | val | 16.4 (.999) / **16.3** (.9993) / 16.4 (.9995) / 18.3 (.9997) | 29.3 (+1.9 extra EMAs) | 2 up blocks/level, one skip each, widths (40, 52, 52); clean, slightly soft |
| exp_034_flow_lean_deep | flow | 983,743 (+16 buf) | 20,000 | 3801 | val | **16.0** (.999) / 16.1 (.9993) / 16.6 (.9995) / 18.1 (.9997) | 32.2 (+1.0 extra EMAs) | 3+3 blocks/level, one skip each, widths (32, 44, 44); clean, slightly smoother |

## Stage 1 smoke test (turn 002, 200 updates each, both concurrently on the MIG slice)

| exp | compile + 1st update | steady speed | peak GPU mem | generate(50) | loss @200 |
|---|---|---|---|---|---|
| exp_010_ddpm_strong | 54 s | 11.1 upd/s | 416 MiB | 5.4 s (100 steps) | 0.102 |
| exp_011_flow_strong | 55 s | 11.1 upd/s | 416 MiB | 3.6 s (25 steps) | 0.315 |

Estimate per full run (concurrent): compile ~1 min + 20k updates / 11 upd/s ≈ 30 min + 4 EMA evaluations
(1,000 samples + KID each; DDPM ≈ 2.5 min each, flow ≈ 1.5–2 min each) ≈ **DDPM ~42 min, flow ~38 min**.
Launched 17:32. **Actual**: 12.8 upd/s each (26.4 min for 20k), training done 18:00, all EMA scores by 18:05 (DDPM 32 min total, flow 30 min).

## Decisions

- Turn 002: killed leftover runs exp_001–004 from turn 001 (baselines come from the sheet).
- Stage 1 network: ADM-style U-Net, widths 32/48/48 at 32/16/8 px, 2 res blocks per level down, 3 up
  (ADM skip layout incl. downsampled maps), attention at 16x16 and 8x8 plus in the middle, 981,667 values.
  A 4th (4x4) level and wider 8x8 levels did not fit under 1M with this block layout; widths kept even
  rather than dropping 16x16 attention (brief: "16x16 if it fits").
- Time embedding: 32-dim sinusoidal of 1000·t → 2-layer MLP (64), FiLM scale/shift on the second GroupNorm.
- Extra EMA evaluation: `train()` keeps the non-primary EMA weights in a module-level `EXTRA_EMA` dict;
  `runner.py` scores each with the fixed `image_score()`/`generate()` into `runs/<run>/ema<decay>/`
  and logs one extra `index.jsonl` row per decay (tag `ema<decay>`). `run()` is unchanged. Primary
  (copied into the model, scored by `run()`) = 0.9998.
- `tools/splice.py` now optionally replaces `train()` too (`tools/train/ema.py`); `tools/smoke.py` = smoke test.

- Timestamped logs: runs are launched through `logrun.sh` (wraps `runner.py`, prefixes each log line with the
  time) so that minutes per 1k updates can be read off the logs. Runner itself is unchanged.
- `pypdf` installed into the venv to read the sheet (no `pdftotext` on the node).

- Prepared (not run) for Stage 2: `exps/exp_012_flow_cosine.py` = Stage 1 recipe with peak LR 2e-3 and
  half-cosine decay to 0 after the 500-update warm-up (`tools/train/ema_cosine.py`); primary EMA 0.9995
  because LR annealing reduces the need for long averaging. Same network. Task is set to whichever wins Stage 1.
- `tools/check_fixed.py` diffs the fixed sections of every experiment against the starter (all OK so far).

- Stage 1 → continue with **flow** (14.4 vs 100.9; not a tie). DDPM's failure looks like epsilon-prediction
  error amplified at the high-noise end of the 100-step cosine schedule (global colour casts); fixable with
  v/x-prediction inside `forward`, but flow is 7x better now, so the GPU goes to flow (brief §6).
- EMA 0.9999 / 0.9998 were contaminated by the random initial weights (0.9999^20000 ≈ 13.5 %, 0.9998^20000 ≈ 1.8 %).
  Fix stacked into exp_012: EMA weight max(1−decay, 1/(step+1)) = running mean until the 1/(1−decay) horizon.
- exp_012_flow_cosine smoke-tested for 100 updates (13.7 upd/s alone, no errors); not launched (next turn).

- Turn 003: standing rule — no AI tool/model/vendor names in tracked files; the agent's own files
  (RESULTS.md, tools/, exps/, logrun.sh, runner.py) were checked and contain none. README.md and p6_brief.md
  (not written by the agent) still contain such names and were left untouched.
- Stage 2 (turn 003): DDPM dropped; flow only. exp_012 launched exactly as prepared.
- exp_013_flow_wide: 012 recipe, widths (48, 56, 56), 1 block down / 2 up per level, attention at 8x8 + middle,
  950,387 values. (48, 64, 64) is 1.15M with this layout (the concatenating up blocks dominate), so the
  64s were trimmed to 56 to land in 0.95–1.0M.
- No third concurrent run: in a three-process smoke test exp_012 ran at 6.0 upd/s vs ~11 with two processes
  (the slice's total throughput is ~fixed, so a 3rd run costs the other two >20 %). Two of the three smoke
  processes also died silently during the concurrent compile (no traceback; not reproduced with two).
  Prepared but not run: exp_014_flow_xpred (012 + x-prediction converted to velocity in `forward`,
  1−t floored at 0.05; 11.6 upd/s in smoke test).
- Prepared follow-up: exp_015_flow_mp = EDM2-style magnitude-preserving U-Net (unit-norm MPConv weights with
  forced weight normalisation after each update, mp_silu / mp_sum / mp_cat, pixel norm in encoder blocks,
  multiplicative time conditioning, no GroupNorm, no biases, zero output gain) in the 012 layout, 927,234
  values, peak LR 1e-2 (EDM2 scale for unit-norm weights). After Stage 2 its schedule was switched to the
  011 one (warm-up then constant; warm-started EMA kept) because cosine did not help. GPU smoke test (300 upd,
  next to exp_012): 12.9 upd/s, 445 MiB, compile 3.6 min under contention, loss falling normally.
- exp_014_flow_xpred rebuilt on the incumbent 011 recipe (lr 1e-3 constant, warm-started EMA) — ready to pair
  with exp_015. Activations stay at
  std ≈ 0.8–1.2 through all blocks at init (CPU check). Layout/widths to be switched to the Stage 2 winner.

- Turn 004: `~/.p6_agent` already existed and holds the absolute path of the CLI the driver launches (checked against the running process); left unchanged. 
- Stage 3 (turn 004): exp_014 and exp_015 launched together exactly as prepared (two processes, ~13 upd/s each;
  two other users' processes also on the slice). Paired single-batch losses (same seed → same batches/noise)
  at 2k/4k/6k/8k: 011 .2045/.2069/.2039/.2152, 015 .1955/.2075/.1934/.2090 (≈2.5 % lower), 014
  .2956/.2643/.2337/.2490 (higher; the 1/(1−t) conversion amplifies errors near t→1). 014 was not killed
  because the reviewer asked for both results and its gap was shrinking.
- Follow-up (a) = exp_017_flow_mp_isqrt: 015 (better mid-training loss) + EDM2's inverse-square-root LR decay,
  lr = 1e-2·min(1, sqrt(T_REF/step)), T_REF = 1500 (≈7.5 % of training, the same proportion as EDM2's t_ref),
  so lr ends at 2.7e-3. Reason: with forced weight normalisation the weight norms cannot grow, so the
  effective step size never shrinks by itself; EDM2 pairs MP nets with this decay. 927,234 values; fixed parts OK.
- Follow-up (b) = exp_016_flow_lrtail: 011 net, lr 1e-3 constant after warm-up, then linear decay to 0 over the
  last 20 % (updates 16k–20k), warm-started EMA, primary EMA 0.9995. 981,667 values; fixed parts OK.
  Neither follow-up has been GPU smoke-tested (a third process would have slowed the live runs); they change
  only the LR lambda, which was checked numerically.

- Turn 005 rulings (reviewer): autoguidance not allowed (changes the effective sampler); augmentation beyond
  flips not allowed (risks counting as extra data); Muon allowed as ONE optimiser class with one step() per update.
- Stage 4 (turn 005): exp_017 and exp_016 launched together exactly as prepared (19:32). The node was heavily
  loaded (load ~24), so compilation took ~5 min.
- exp_018_flow_muon (prepared, smoke-tested, not run): 011 net and recipe (warm-up then constant LR,
  warm-started EMA), with the optimiser replaced by `MuonAdamW`, a single `torch.optim.Optimizer` subclass.
  Groups marked muon=True (83 hidden weight matrices: block convs, 1x1 skips, attention qkv/proj, FiLM linears;
  963,328 values) get Nesterov momentum 0.95, then 5 quintic Newton-Schulz steps in bf16 (conv kernels
  flattened to out x in*k*k), lr 0.02·sqrt(max(1, rows/cols)). The other groups (biases, GroupNorm gains,
  time-embedding MLP, input and output convs) get AdamW at lr 1e-3, betas (0.9, 0.99), all in the same step().
  torch 2.14 ships `torch.optim.Muon`, but it is 2-D-only and a separate optimiser, so the one-class version
  is self-contained instead. CPU checks: the AdamW branch matches `torch.optim.AdamW` to 6e-8; Newton-Schulz
  singular values are 0.67-1.13 (the intended band for these coefficients). The Newton-Schulz step is batched
  over matrices of the same shape: 33 vs 87 ms per optimiser step (torch AdamW: 5 ms), timed with the two
  live runs on the slice.
- Muon smoke test (300 updates, seed 3801, run next to exp_016/017, so speeds are contended): paired with a
  300-update 011 reference on identical batches, loss at 300 was 0.240 vs 0.264 and the mean of the logged
  losses over 200-300 was 0.256 vs 0.289 (-11 %). Stable: no spikes, the largest loss is update 1's.
  Speed was 5.3 vs 8.7 upd/s (unbatched version; -39 %); the batched step should close part of the gap.
- Prepared for turn 006 (not run): exp_019_flow_lr2e3 = 011 net and recipe (warm-up, constant LR, warm-started
  EMA) at constant lr 2e-3. It isolates the higher LR that 012 confounded with cosine decay. Recommended
  pairing: exp_018_flow_muon + exp_019_flow_lr2e3 (both on the 011 net, so both compare cleanly with 14.4).
- Muon references: K. Jordan et al. 2024, "Muon: An optimizer for hidden layers in neural networks" (blog,
  kellerjordan.github.io/posts/muon); J. Liu et al. 2025, "Muon is Scalable for LLM Training", arXiv:2502.16982;
  F. Schaipp 2025, "Optimization Benchmark for Diffusion Models on Dynamical Systems", arXiv:2510.19376.

- Turn 006 (reviewer): late LR decay dropped. exp_019_flow_lr2e3 is superseded and was never run (012 already
  contained 2e-3 and lost ~3.6 KID, against ~1.3 for decay alone); the file is kept for the record. Launched
  exp_018_flow_muon (as smoke-tested) and exp_019_flow_dropout together at 20:11.
- exp_019_flow_dropout: 011 net and recipe plus nn.Dropout(0.1) in every residual block, after the second
  activation and before conv2 (Ho et al. 2020; EDM). Eval mode at sampling: image_score() calls model.eval()
  right before generate(), and nothing in run() or the runner's extra-EMA scoring switches it back; the
  extra-EMA models are fresh and also go through image_score(). CPU check: eval outputs are deterministic,
  train outputs stochastic. Its training losses are not paired with 011's: the dropout masks draw from the
  same CUDA random stream as the noise.
- Muon speed in the full run: 2,000 updates in 2:47 vs 2:36 for 011 (batched Newton-Schulz), about 7 % slower.
- Prepared (a) exp_020_flow_muon_dropout = 018 + 019 (the case where both help); 981,667 values.
- Prepared (b) exp_021_flow_hires: 011 recipe, widths (40, 48, 40) instead of (32, 48, 48), 958,199 values.
  Reason: the 32-px level (2 down + 3 up blocks) is the narrowest and carries the fine texture that KID is
  sensitive to; standard CIFAR U-Nets (DDPM, ADM, DDPM++) never make the full-resolution level narrower than
  the others. The 8x8 level already has global attention, so it can give up width. Depth and 16x16 attention
  are kept (013 showed they matter). Cost: about 1.5x the compute at 32 px, so it trains slower.
- runner.py now also saves the trained weights (primary + each extra EMA) to runs/<run>/weights/*.pt
  (git-ignored), for offline diagnostics only. New tools/sampler_probe.py (analysis only) re-samples saved
  weights with Euler at 25/50/100 steps using the same noise seed and KID code, to measure how much KID the
  fixed 25-step sampler costs. The first runs with saved weights are next turn's (018/019 started earlier).
- Muon paired losses vs 011 (identical batches): about equal to 6k, then 4-9 % higher from 8k to 18k
  (e.g. 12k .1818 vs .1661). The 300-update smoke advantage did not last.
- exp_020_flow_muon_dropout will not be run: neither 018 nor 019 helped, so its condition (both help) failed.
- Prepared exp_022_flow_muon_rms (not recommended first): 018 with the Liu et al. 2025 update scaling
  0.2·sqrt(max(rows, cols)), which matches AdamW's update RMS so the Muon group shares the AdamW lr 1e-3, plus
  weight decay 0.1 on the Muon group. Checked: Muon update RMS = 0.17·lr. Reason: Jordan's lr 0.02 gives
  updates about 5x AdamW's RMS on these small matrices; this matches 018's drifting weights and collapsed
  long EMAs.

- Turn 007 (reviewer): sampler probe approved as analysis only (never used to choose anything inside the
  final file); Muon retry 022 deferred. Launched exp_021_flow_hires and exp_023_flow_noflip together at 20:50.
- exp_023_flow_noflip: the 011 net on the current recipe (warm-started EMA, primary EMA 0.9995) with the
  horizontal-flip line removed. It differs from 021 only in widths and flips. The warm start affects only the
  long EMAs, so its 0.999/0.9995 scores compare directly with 011's. Removing the flip also removes one CUDA
  random draw per update, so its losses are not paired with 011's.
- Prepared (a) exp_025_flow_hires_noflip = 021 + 023, for the case where both help.
- Prepared (b) exp_024_flow_beta2: 011 net and current recipe with AdamW betas (0.9, 0.999) instead of
  (0.9, 0.99). Reason: it is the setting DDPM/iDDPM/ADM trained CIFAR with; a longer second-moment average gives
  steadier steps at constant LR (a tighter fit around the same minimum) without changing the LR scale.
  Rejected: fp32 (bf16 rounding of ~2^-8 relative on |v| ~ 1.4 is an MSE floor of ~3e-5 against a loss of 0.18,
  so it is negligible) and clip off (a no-op unless gradient norms exceed 1; tools/gradnorm_probe.py measures
  this on saved weights).
- Tooling bug (mine, turn 006 change): the runner's weight capture patched the dict returned by
  runpy.run_path, which is a COPY of the program's globals, so run() never used the wrapper. The first 023 run
  crashed right after its primary score (EMA 0.9995: 15.27, from the log; no index row), losing the
  extra-EMA scores and the weights. 021 (at 14k updates) had loaded the same runner, so it was stopped. Fixed:
  the wrapper goes into run.__globals__, and weight saving now happens after the index row and is never
  fatal. Verified end to end with a fake program in a scratch directory (no index row written). Both
  runs were relaunched at 21:22 with the fixed runner; the partial outputs are kept as *_crashed / *_killed.
  tools/sampler_probe.py had the same copy problem (fixed; checked that 25/50/100 steps give 25/50/100 model
  calls). Cost: ~30 GPU-minutes.
- tools/sampler_probe.py had a second bug: image_score is wrapped by torch.no_grad, so its __globals__ are
  torch's, and the first probe (all three step counts 15.32) silently sampled with 25 steps every time.
  Fixed: it patches run.__globals__ instead, checked through image_score itself with a counting dummy model
  (500 / 2000 calls for 25 / 100 steps). The runner fix was unaffected (run() is not decorated); the probe's
  25-step score matches the runner's (15.32 vs 15.33), confirming the saved weights are the trained model.

## Sampler probe (analysis only; EMA 0.9995 weights; same noise seed and KID code as image_score)

| exp | Euler 25 (fixed) | Euler 50 | Euler 100 | 25 -> 100 |
|---|---|---|---|---|
| exp_021_flow_hires | 16.03 | 14.27 | 13.04 | -19 % |
| exp_023_flow_noflip | 15.32 | 14.72 | 14.44 | -6 % |

Gradient norms at the end of training (50 batches, tools/gradnorm_probe.py): 021 median 0.045 (max 0.086),
023 median 0.059 (max 0.123); never above the clip threshold of 1.0, so clipping is inactive after warm-up.

## Real-data floor (turn 008; analysis only, nothing in any experiment or the final file depends on it)

tools/real_floor.py scores the first 1,000 images of image_data("train") (its fixed seed-3802 order, no flips)
against the validation split through image_score() itself (generate() swapped to hand out those images, 50
per call), so pixels, Inception features and KID settings are identical to every run: **KID x1000 = -0.02**.
A perfect memoriser of the training set would score about 0, so the metric itself does not rule out single
digits. Our 14.4 is model error, not a property of the train/validation split.

- Turn 008 (reviewer): Muon dropped (022 not to be run); exp_025 (a) not run. Launched exp_024_flow_beta2 and
  exp_026_flow_lowres together at 22:10, both with the finer EMA grid 0.999 / 0.9993 / 0.9995 / 0.9997 / 0.9998
  (primary 0.9995).
- exp_026_flow_lowres: 011 recipe; the Model now accepts blocks per level. Widths (32, 40, 40, 40) at 32/16/8/4 px,
  blocks (1, 2, 2, 2), attention at 16/8/4 (the middle blocks now sit at 4x4), 944,771 values. Reason: 013 showed
  depth beats width in this budget, and a 4x4 level roughly doubles the processing depth at global scale
  (shape), at the cost of one 32x32 down block and 48 -> 40 at 16/8. Widening 8x8 instead is costly: (32, 48, 64)
  is 1.26M even with a single 32x32 block, because every up block concatenates skips. Smoke test (200 updates,
  next to 024's): 18.1 upd/s, 358 MiB, all extra EMAs load.
- Prepared (a) exp_028_flow_lowres_beta2 = 026 + beta2 0.999 (944,771 values), for the case where both help.
- Prepared (b) exp_027_flow_convnext: ConvNeXt-style blocks (Liu et al. 2022) in the 011 skeleton: 7x7 depthwise
  conv -> GroupNorm with FiLM scale/shift -> 1x1 (4x) -> SiLU -> zero-initialised 1x1, residual; 1x1 skip when the
  width changes; same attention. Widths (48, 56, 64), 2 blocks per level, 935,715 values. Reason: the 7x7
  depthwise kernels give a much larger receptive field at every level (global shape), and a block costs about
  8C^2 + 49C against 18C^2 for the two 3x3 convs, so the same budget buys about 1.3x the width at the same
  depth. adaLN-Zero gating was rejected: the residual outputs are already zero-initialised, so it should add little.
- 024 paired losses vs 011 (identical batches): +0-2 % to 6k, then the same within 0.5 % from 12k to 16k.
- (a) exp_028_flow_lowres_beta2 will not be run: 026 lost and 024 only tied.
- exp_027_flow_convnext GPU smoke test (300 updates, alone on the slice): compile 101 s, 11.6 upd/s, 854 MiB,
  all extra EMAs load, stable; mean logged loss 200-300 = 0.286 (011's 300-update reference: 0.289, not
  paired because the initialisation differs).
- Prepared exp_029_flow_convnext_deep: the same ConvNeXt blocks, deeper and narrower: widths (40, 56, 56),
  3 blocks per level, expansion 3, 895,511 values (not smoke-tested; it only changes sizes in the 027 code).

- Turn 009 (reviewer): 029 smoke-tested (200 updates, alone: 10.1 upd/s, 863 MiB, stable, extra EMAs load), then
  exp_027_flow_convnext and exp_029_flow_convnext_deep launched together at 22:46 with EMA grid
  0.999 / 0.9993 / 0.9995 / 0.9997 (0.9998 dropped, never best).
- Prepared (a) exp_030_flow_precond: EDM-style preconditioning (Karras et al. 2022) for the velocity target,
  inside Model.forward, on the 011 network (neither ConvNeXt result was in yet). Convention note: the fixed
  flow_path is x_t = t*data + (1-t)*noise (t = 1 is data), the reverse of the formula in the instruction, so the
  coefficients are derived for the code's convention. With d = t^2 s^2 + (1-t)^2: c_in = 1/sqrt(d),
  c_skip = (t s^2 - (1-t))/d, c_out = s/sqrt(d), the variance-minimising choice that gives F a unit-variance target
  (derivation: Var(v - c_skip x_t) = s^2/d at the optimum). d >= s^2/(1+s^2) > 0, so nothing divides by (1-t).
  s = SIGMA_DATA = 0.5 (measured once: pixel std of the 4,000 training horses = 0.4985; hard-coded).
  CPU checks on real training images: Var(c_in x_t) = 1.000-1.007 and Var(F target) = 1.000-1.007 at every t in
  0..1 (the excess comes from the -0.06 image mean); at init the net outputs exactly c_skip*x_t, the best
  linear predictor (MSE 0.25 / 0.80 / 1.08 at t = 0 / 0.5 / 0.96 vs 1.25 for a zero output). Time embedding unchanged.
- Prepared (b) exp_031_flow_attn32: 011 + one head-32 self-attention layer in the last 32x32 encoder block
  (down level 0, block 1), 985,955 values, no trimming needed (+4.3k).
- GPU smoke tests after 027/029 finished (each alone, 300 updates): 031_flow_attn32 18.8 upd/s, 767 MiB vs
  030 (011's network plus a few elementwise ops; the speed reference) 18.2 upd/s, 416 MiB, i.e. no slowdown
  (threshold 0.6x). 030_flow_precond paired with 011 (same initialisation, batches and noise): mean logged loss
  0.365 vs 0.289 over updates 200-300 (+26 %), then 0.240 vs 0.227 over 667-1000 (+6 %, 1,000-update runs). It
  starts lower (first-batch loss 0.80 vs 1.27) but learns more slowly at first; the gap is closing. Not a stop
  signal, but no early evidence of a gain either.

- Turn 010: exp_030_flow_precond and exp_031_flow_attn32 launched together at 00:07 with
  --snapshots 10000,15000. Snapshot mechanism (analysis only, must not be in the final file): the programs
  carry an inert hook `SNAPSHOT_AT, SNAPSHOTS = (), {}`; train() copies the primary EMA to CPU when
  step in SNAPSHOT_AT. Only the runner sets SNAPSHOT_AT (via run.__globals__, flag --snapshots), then scores
  each snapshot with the same image_score() path as an extra index row tagged snap<step>_ema0.9995. Verified
  end to end with a fake program in a scratch directory (snapshots at steps 10/15 returned exactly those values).
- 030 paired losses vs 011 (identical batches): +0.5-1 % at 2k-8k and equal at 20k (0.1803 vs 0.1797), yet KID
  was much worse. Same mean velocity error, but distributed differently over t and pixels.
- 031 ran at ~9.5 upd/s next to 030 (alone: 18.8, same as 011). 32x32 attention is cheap alone but
  contends for memory bandwidth.
- (a) exp_032_flow_precond_attn32 (030 + 031, 985,955 values) prepared but not to be run: neither helped.
- (b) exp_033_flow_lean (capacity, because the 10k/15k/20k curve is flattening): 011 recipe with
  2 decoder blocks per level instead of 3, each taking one encoder-block skip (no input-conv or
  downsampled-map skips). That frees about 19 % of the parameters, spent on widths (40, 52, 52) instead of
  (32, 48, 48): 984,731 values. GPU smoke test (200 updates, alone): 20.2 upd/s, 396 MiB, stable, extra EMAs load.
- Second capacity point exp_034_flow_lean_deep: the same one-skip-per-block decoder used for depth instead of
  width: 3 encoder + 3 decoder blocks per level, widths (32, 44, 44), 983,759 values (CPU-checked only).

- Turn 011: 034 smoke-tested (200 updates, alone: 17.7 upd/s, 460 MiB, stable), then exp_033_flow_lean and
  exp_034_flow_lean_deep launched together at 00:52 (EMA grid .999/.9993/.9995/.9997, no snapshots).
- Final pipeline prepared (no GPU run yet): final/lab06_final.py = exp_011 cleaned. Single EMA 0.9995 (warm-started,
  i.e. a running mean for the first 2,000 updates, so the initial weights leave no bias) copied into the model after
  training. The EXTRA_EMA machinery, the snapshot hook and the dead DDPM branch of train() were removed;
  STUDENT_ID = "xmzj54"; __main__ calls plain run(). Header (comments only, above the starter's first line):
  abstract (8 lines, no KID claims), 9 verified references (arXiv ids and venues checked), and the exact line
  `# AI acknowledgement: AI_ACK_PLACEHOLDER`. Training RNG use is identical to the warm-started-EMA experiments, and
  011's 0.9995 EMA (14.4) is equivalent: the old init bias at 0.9995 is 0.9995^20000 ~ 5e-5.
- tools/check_fixed.py now allows a header of comment/blank lines before the starter's first line; everything from
  that line on is still compared byte for byte (negative tests: a non-comment header line, or one changed fixed line,
  are both reported DIFFERS). Result for final/lab06_final.py: OK. tools/count.py: 981,667 values.
- tools/final_smoke.py: train() smoke test that records every EMA update and checks the weights
  max(1 - EMA_DECAY, 1/(step+1)), that the model ends up holding exactly the EMA (not the raw weights), and that
  generate() is finite. The CPU attempt was stopped: CPU compile took over 2 of our 4 cores and slowed the GPU runs;
  it is run on the GPU after 033/034 finish.
- final/lab06_final.py smoke test on the GPU after 033/034 finished (tools/final_smoke.py, 100 updates): passed.
  EMA update weight exactly 1/(step+1) at every update; the model ends up holding exactly the EMA, not the raw
  weights; generate() finite. First-batch loss 1.268, the same as 011's 1.2677, so it follows the same RNG path.
  (The first attempt flagged 200 lerp calls: AdamW's foreach path also calls torch._foreach_lerp_ for its first
  moment; the test now counts only calls on the model's own parameters.)

- Turn 012 (reviewer approved final/lab06_final.py). Submission copy final/submit/lab06_final.py (git-ignored
  directory, checked with check-ignore): only the acknowledgement placeholder line was replaced, by the reviewer's
  wording with the names filled in (5 comment lines, max 110 chars). diff vs final/lab06_final.py is a single hunk at
  line 21 (placeholder out, 5 comment lines in); check_fixed OK; 981,667 values. data symlink added; run once at
  01:36 as `python lab06_final.py` from final/submit (plain run(), seed 3801), log in final/final_run.log.
- exp_035_flow_lr6e4: 011 net and recipe with constant lr 6e-4 (same 500-update warm-up), EMA grid
  .999/.9993/.9995/.9997; launched 01:36 alongside the final run.
- Prepared exp_036_flow_powerema (not run): EDM2 power-function EMA, beta_t = (1 - 1/t)^(gamma+1), gamma = 6.94,
  so that sigma_rel = sqrt((g+1)/((g+2)^2 (g+3))) = 0.10 matches fixed 0.9995 at t = 20k (std of its averaging
  profile = 1/(1-0.9995) = 2,000 updates = 0.10 t). The fixed decays are tracked in the same run for an exact,
  paired comparison. Expected: a tie. At t = 20k the two profiles nearly coincide (std 2,000 vs 2,000 updates,
  mean age 2,237 vs 1,999; weight on updates before 15k 10 % vs 8 %), and with a constant LR the late iterates
  are roughly stationary, so both average the same window. EDM2's gains come from post-hoc profile choice and
  the interaction with LR decay, neither of which applies here.

## Stage conclusions

**Stage 1 (turn 002).** Strong recipe at 20k updates: flow 14.4 (EMA 0.9995) — already below the lecturer's
30.6 — vs DDPM 100.9 (EMA 0.9998). Shorter EMA (0.999–0.9995) wins under a constant LR; long EMAs were
broken by init bias (fixed for Stage 2). Flow is the task from here on.

**Stage 2 (turn 003).** Neither change beat the incumbent exp_011 (14.4). Peak LR 2e-3 + cosine decay (012)
gave 18.0 (+25 %, beyond the ~10 % tie band); the wide/shallow net on the same recipe (013) gave 20.8 vs 012's
18.0 (+15 %): the deeper 32/48/48 layout with 16x16 attention is the better use of the budget. The warm-started
EMA fixed the init-bias problem (012 EMA 0.9998: 18.8 vs 25.9 before) but 0.9999 is still worse, and
0.999–0.9995 remain best. Incumbent: exp_011 (constant lr 1e-3, EMA 0.9995). Speed: the wide net trains
~30 % faster (17 vs 12.8 upd/s).

**Stage 3 (turn 004).** The EDM2 magnitude-preserving U-Net (015) gave 14.8 vs the incumbent's 14.4, a tie
(<10 %). It used 6 % fewer parameters and trained slightly faster, and its long EMAs held up much better
(0.9998: 16.0 vs 25.9; 0.9999: 21.5 vs 355). It also had ~2.5 % lower paired training loss. x-prediction
converted to velocity (014) was clearly worse (29.7, +106 %), with blotchy textures: at this tiny capacity and
in pixel space (3,072 dims), the 1/(1−t) factor amplifies the network's errors near the data end, where the
fixed loss and 25-step sampler need precise velocities. Li & He's gains come from high-dimensional patches,
which don't apply here. Dropped. Incumbent stays exp_011 (simpler on a tie). The MP net is the base for
schedule work, because its native EDM2 schedule (inverse-sqrt decay) has not been tried yet.

**Stage 4 (turn 005).** Neither schedule follow-up beat the incumbent. The short linear tail on 011 (016) gave
15.7 vs 14.4, a tie (+9 %); its paired loss was 0.3-1.5 % lower after the decay, but KID did not improve.
EDM2's inverse-sqrt decay on the MP net (017) was clearly worse than constant LR (24.3 vs 14.8, +64 %), with
grainy textures, though its paired training loss matched 015's. Across 012, 016 and 017, every late LR
reduction gave equal or worse KID at equal or lower loss: on this budget, EMA over a constant-LR trajectory
beats annealing. Decay is dropped. The open question is the opposite direction (a higher constant LR, 019),
plus the optimiser (Muon, 018). Incumbent unchanged: exp_011 (14.4).

**Stage 5 (turn 006).** Muon (018) was clearly worse (34.2 vs 14.4) with muddy samples. Its long EMAs collapsed
(80, 223), i.e. the weights were still moving fast at the end, consistent with update steps about 5x
AdamW-sized. Dropout 0.1 (019) tied the incumbent exactly (14.4 at EMA .999) with slightly smoother samples;
on a tie the simpler option wins, so no dropout. Incumbent unchanged: exp_011 (14.4). Now 015 (MP net),
016 (LR tail) and 019 (dropout) all landed at 14.4-15.7 whatever their training loss; this looks like a
floor. Prime suspect: the fixed 25-step Euler sampler, plus about +-2 KID of 1,000-sample noise. The
sampler probe will measure it on the next run's saved weights.

**Stage 6 (turn 007).** Neither tighter-fit change helped. No flips (023): 15.3 vs 14.4 (+6 %, tie, leaning
worse); flips are kept. Widths (40, 48, 40) (021): 15.6 at EMA .999 (+8 %, tie, leaning worse), with crisper
detail but less coherent shapes. Sampler probe: 4x more Euler steps lowers KID by only 6 % (023) to 19 %
(021), i.e. 1-3 KID. So the fixed 25-step sampler costs something, more for the sharper wide-32px model, but
it is not the main floor; the variants keep landing at 14-16 for model/data reasons, within the
1,000-sample KID noise. Clip at 1.0 never activates, so "clip off" would be a no-op. Incumbent unchanged:
exp_011 (14.4). Eight changes in a row have tied or lost against it.

**Stage 7 (turn 008).** Real-data floor: 1,000 real training horses score -0.02 against validation, so the
metric allows far lower than our 14.4; the gap is model error. beta2 0.999 (024) tied (14.2 vs 14.4, -1 %).
Its finer EMA grid gives a smooth optimum at 0.9995 (15.2 / 14.6 / 14.2 / 14.7 / 16.8 for .999 to .9998), so
0.9995 is confirmed and the old 0.9998 'cliff' (25.9 in 011) was mostly init bias plus a slope after .9997.
Shifting capacity to a 4x4 level (026) was clearly worse (19.3, +34 %) with blurrier samples: like 021
(capacity to 32 px, 15.6), moving width away from 011's balanced 16/8-px levels hurts. Incumbent unchanged:
exp_011 (14.4); 024 is an equal alternative.

**Stage 8 (turn 009).** ConvNeXt-style blocks did not beat the 3x3 residual blocks: 027 (wide, 2 per level) 16.0
and 029 (deep, 3 per level) 16.3, both at EMA 0.999 (+11 % and +13 %, just beyond the tie band), with clean but
slightly soft samples. They also cost ~2.3x 011's GPU time (depthwise 7x7 + 4x MLPs at 32 px are bandwidth-bound
on this slice). Block type, layout (013/021/026), optimiser (018/024), regularisation (019/023) and schedule
(012/016/017) have now all tied or lost against exp_011 (14.4), and the real-data floor (-0.02) says the gap is
model error. The two untried directions are input/output scaling (030, preconditioning) and full-resolution
attention (031), both on the 011 network at 011's speed.

**Stage 9 (turn 010).** EDM-style preconditioning (030) was clearly worse (19.6, +36 %) at equal training loss,
with softer, partly washed-out samples. 32x32 attention (031) was worse (16.1, +12 %). Training curve of the
EMA 0.9995 weights at 10k / 15k / 20k: 030 28.5 / 21.2 / 19.6 and 031 24.3 / 17.6 / 16.1. KID falls about 27 %
from 10k to 15k but only 7-8 % (~1.5 KID, inside the noise band) from 15k to 20k, so the curve is flattening
fast: we are mostly capacity/architecture-limited, not budget-limited. Incumbent unchanged: exp_011 (14.4).

**Stage 10 (turn 011).** Spending the third up-block's parameters elsewhere did not help: 033 (2 up blocks/level,
wider) 16.3 and 034 (3+3 blocks with one skip each, deeper) 16.0, both +11-13 %, just beyond the tie band. Both
removed ADM's extra skips from the input conv and the downsampled maps; with this and 013/021/026/027/029/031, every
reallocation of 011's ~1M parameters has tied or lost. Incumbent unchanged: exp_011 (14.4). The final pipeline is
ready: final/lab06_final.py = exp_011 cleaned, checks OK, smoke test passed.
