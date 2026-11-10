# Practical 6 — results log (xmzj54)

## Environment (recorded 2026-11-10, turn 001)

| Item | Value |
|---|---|
| Host | gpu10 (NCC, SLURM job, 24 h limit) |
| GPU | NVIDIA A100 80GB PCIe, **MIG slice 1g.10gb**: 14 SMs (1/7 of the A100), 9.7 GiB memory |
| Driver / CUDA | 610.43.02 / CUDA 13.3 UMD; torch built for CUDA 13.0 |
| CPU | 4 cores allocated (SLURM cpu=4), 20 GB RAM allocated |
| Python | 3.12.3 (`~/deep_learning` venv) |
| torch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 |
| torch-fidelity | 0.4.0 (Inception weights cached) |

## All runs

KID x 1000 on 1,000 samples, validation split unless stated. Minutes = runner wall clock (train + sample + KID).

| exp | task | params | updates | seed | split | KID x1000 | minutes | notes |
|---|---|---|---|---|---|---|---|---|

## Decisions

- Timestamped logs: runs are launched through `logrun.sh` (wraps `runner.py`, prefixes each log line with the
  time) so that minutes per 1k updates can be read off the logs. Runner itself is unchanged.
- `pypdf` installed into the venv to read the sheet (no `pdftotext` on the node).

## Stage conclusions
