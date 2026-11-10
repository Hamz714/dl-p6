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
