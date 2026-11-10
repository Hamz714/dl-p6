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
  values, peak LR 1e-2 (EDM2 scale for unit-norm weights) with the 012 schedule. Activations stay at
  std ≈ 0.8–1.2 through all blocks at init (CPU check). Layout/widths to be switched to the Stage 2 winner.

## Stage conclusions

**Stage 1 (turn 002).** Strong recipe at 20k updates: flow 14.4 (EMA 0.9995) — already below the lecturer's
30.6 — vs DDPM 100.9 (EMA 0.9998). Shorter EMA (0.999–0.9995) wins under a constant LR; long EMAs were
broken by init bias (fixed for Stage 2). Flow is the task from here on.
