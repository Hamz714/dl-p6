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

## Stage 1 smoke test (turn 002, 200 updates each, both concurrently on the MIG slice)

| exp | compile + 1st update | steady speed | peak GPU mem | generate(50) | loss @200 |
|---|---|---|---|---|---|
| exp_010_ddpm_strong | 54 s | 11.1 upd/s | 416 MiB | 5.4 s (100 steps) | 0.102 |
| exp_011_flow_strong | 55 s | 11.1 upd/s | 416 MiB | 3.6 s (25 steps) | 0.315 |

Estimate per full run (concurrent): compile ~1 min + 20k updates / 11 upd/s ≈ 30 min + 4 EMA evaluations
(1,000 samples + KID each; DDPM ≈ 2.5 min each, flow ≈ 1.5–2 min each) ≈ **DDPM ~42 min, flow ~38 min**.
Launched 17:32.

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

## Stage conclusions
