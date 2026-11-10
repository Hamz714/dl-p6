# Practical 6 campaign brief — COMP3801, CIFAR-10 horses, DDPM or image flow matching

Student ID: `xmzj54`. Deadline: **Fri 13 Nov 2026, 14:00 UK**. The final zip must exist by **Thu 12 Nov, 22:00 UK**.

## 1. Goal

Lowest possible **validation KID x 1000** for the final submission, reached with well-implemented,
state-of-the-art methods and code that stays clean and simple (simplicity is also marked).

| Reference point | KID x 1000 |
|---|---|
| Starter DDPM (4k updates) | 220.0 |
| Starter flow (4k updates) | 202.9 |
| Lecturer's example inside the constraints | 30.6 |
| Our target | as low as possible; < 20 good, single digits stretch |

Mark bands: pass = small gain; 2:1 = significant gain with course techniques; high first = substantial gain
with well-implemented state-of-the-art methods from recent papers, high-quality samples, inside the budget.

## 2. How you are being run (read carefully)

- You run **headless, one turn at a time**, launched by `bridge/driver.sh`. Each turn's instruction comes from
  a reviewer (another Claude instance working with Hamzah) as `bridge/inbox/NNN.md`.
  Your **final message** of the turn is saved as `bridge/outbox/NNN.md`; the driver commits and pushes the
  whole repo to GitHub, where the reviewer reads it.
- **Nobody answers questions mid-turn.** Never stop to ask. Take the option you would mark "(Recommended)",
  carry on, and log it in `RESULTS.md` under **Decisions** with a one-line reason.
  If something genuinely needs a human ruling (an assignment-rule question), finish everything else and put it
  under `QUESTIONS FOR REVIEWER` at the top of your final message.
- **Do not run git** — the driver commits and pushes every few minutes and at the end of each turn.
- Work only inside `~/p6`. Never delete or overwrite `runs/`, `RESULTS.md` history or other experiments.
- **One stage per turn.** Launch the stage's runs, **wait until they finish** (poll with `sleep` of at most
  9 minutes per command), then report. Keep a turn under ~3 h; if a stage would take longer than ~4 h, split
  it and report partial results.
- Run several experiments **concurrently** on the GPU when they fit (these models are small).
  Use `nohup python runner.py ... > logs/<name>.log 2>&1 &` and poll the logs.
- Final message format (≤ 40 lines): what ran; a table (exp, task, params, updates, seeds, KID mean ± sd,
  minutes); conclusion; recommended next stage; questions (if any). Put sample-quality notes in one line.
- Keep `RESULTS.md` up to date: environment, a table of **every** run, Decisions, and per-stage conclusions.
  Save `samples/batch-01.png` of every run (the runner keeps it in `runs/<name>/samples/`).

## 3. The rules (from the sheet — treat as hard)

The starter is `starter/dl-practical6.py`; the sheet is `starter/dl-practical6.pdf` (section 8 and 9).

**Fixed — must stay byte-identical to the starter** (check with a diff at the end of every stage):
everything in the `# %% Data, evaluation and submission` section (`program_source`, `model_size`,
`save_submission`, `show_results`, `image_data`, `image_bytes`, `Images`, `image_score`,
`diffusion_schedule`, `add_noise`, `diffusion_loss`, `flow_path`, `flow_loss`, `sample_ddpm`, `sample_flow`),
the `MAX_PARAMETERS...` line, `generate()` (100 DDPM steps / 25 Euler steps), and `run()`.

**Editable:** `STUDENT_ID`, `CONFIG`, the model classes and `build_model()`, and `train()` (optimiser,
LR schedule, warm-up, EMA, augmentation, mixed precision, compile, gradient clipping), plus new helper
functions/classes placed in the model/training sections.

**Hard limits**
- ≤ 1,000,000 parameters + buffer elements in the model that `run()` builds and `model_size()` counts.
- ≤ 20,000 optimiser updates; **exactly one optimiser step per update, one batch of ≤ 64 images per update**;
  `CONFIG["steps"]` must equal the real number of updates. One optimiser only.
- Fresh initialisation; train only on the fixed 4,000 training horses.
- No extra data, pretrained models, distillation teacher, gradient accumulation or extra batches per update.
- Inception is for evaluation only — never for training, checkpoint picking or EMA picking inside the final file.
- No seed picking: the final file calls the plain `run()` (seed 3801) exactly once.

**Rulings on techniques**

| Ruling | Techniques |
|---|---|
| Allowed | Any architecture: multi-resolution U-Net, residual blocks, self-attention, adaptive GroupNorm / scale-shift (FiLM/adaLN) conditioning, sinusoidal/Fourier time embeddings, ConvNeXt-style blocks, EDM2 magnitude-preserving layers and forced weight normalisation, dropout, zero-initialised output layers |
| Allowed | Output parameterisation *inside* `Model.forward` (e.g. the net predicts a clean image and `forward` converts it to the epsilon / velocity the fixed loss expects), as long as the loss and sampler code are untouched. Mention it in the abstract |
| Allowed | AdamW settings, LR warm-up and decay, weight decay, gradient clipping, EMA of the weights (kept outside the model like optimiser state, copied into the model after training), random horizontal flips of the training images, bf16 autocast, TF32, `torch.compile`, channels_last |
| Grey — only when the reviewer asks | Guidance inside `forward` (CFG, autoguidance), label dropout, anything that changes the effective number of network evaluations per sampling step |
| Banned | Editing the loss functions, the time-sampling distribution or loss weighting; changing the sampler or its step count; extra updates/optimisers; training on validation/test images; seed or checkpoint picking in the final file |

## 4. Tools provided

- `runner.py` — `python runner.py exps/exp_012_name.py --seed 3801 --split validation`
  Loads a complete experiment program **without** executing its `__main__` block, calls its `run()` with
  `out=runs/<exp>_s<seed>_<split>`, `show=False`, and appends one JSON line to `runs/index.jsonl`
  (exp, seed, split, KID x1000, params, buffers, config, minutes). Use `--split test` for a held-out check
  (the 1,000 CIFAR-10 test horses).
- Every experiment is a **complete copy** of the starter program (`exps/exp_NNN_short_name.py`) with only
  the editable parts changed, so the winner can become the final file directly. Set `STUDENT_ID = "xmzj54"`.
- Python env: `source ~/deep_learning/bin/activate` (torch-fidelity already installed).
  Data: `~/p6/data` (symlinked to `~/data` if that exists).

## 5. Compute policy and selection protocol (Hamzah's standing order — overrides anything below)

The GPU is a small MIG slice of an A100 (1g.10gb), so GPU time is the scarcest resource.
- **Highest expected KID reduction per GPU-minute first, always.** Before launching anything, ask "is this the
  run most likely to cut KID the most right now?" If not, don't launch it.
- **No repeats, no reproduction checks, no extra seeds, no test-split checks** unless the reviewer asks.
  Every run is seed 3801 on validation and tests a new idea.
- **Stack known-good techniques together** instead of ablating them one at a time; ablate only when a change
  looks harmful or the result is surprising.
- Use cheap signals to kill bad runs early: if a run's loss diverges or is clearly worse than the incumbent at
  the same step, stop it and say so.
- KID from 1,000 samples is noisy (roughly ±2–4 at KID ≈ 30), so treat differences under ~10% as a tie and
  prefer the simpler option on ties. Don't spend runs resolving ties.
- Track several EMA decays inside one training run and evaluate each at the end: one training run, several
  answers. That counts as one run.
- Glance at `samples/batch-01.png` for each run and note obvious problems in one line.

## 6. Research plan

The binding constraint is the **training budget** (20k updates × 64 = 1.28M images ≈ 320 epochs of 4k images),
far more than the parameter budget. Favour things that converge fast and improve sample quality per update.

The baselines are known from the sheet (DDPM 220.0, flow 202.9 at 4k updates) — do **not** re-run them.

**Stage 1 — strong recipe straight away** (both tasks, full 20k updates, concurrently)
One "everything known-good" model per task, so the first results are already far below the baselines:
- ADM/DDPM++-style U-Net at 0.9–1.0M params (parameter counter in the file): 32→16→8 (→4) levels,
  2 residual blocks per level, GroupNorm + adaptive scale-shift from a sinusoidal time MLP, self-attention at
  8×8 (and 16×16 if it fits), skip concatenation, zero-initialised final conv and residual output convs.
- AdamW lr ~1e-3 → tune later, betas (0.9, 0.99), ~500-update warm-up, gradient clip 1.0, random horizontal flips,
  bf16 autocast + TF32 (+ `torch.compile` if it compiles quickly), EMA tracked at 0.999 / 0.9995 / 0.9998 / 0.9999.
- Decide DDPM vs flow from this pair; afterwards spend the GPU only on the winning task.

**Stage 2 onwards — next biggest lever each time** (rough expected order of value; re-order on evidence)
1. Learning rate / schedule (constant + EMA vs cosine vs inverse-sqrt decay) on the winner.
2. Architecture capacity allocation: widths/levels/attention placement at the same budget; ConvNeXt-style
   blocks; a simplified EDM2 magnitude-preserving U-Net (MP layers, forced weight normalisation).
3. Output parameterisation inside `forward` (x-prediction converted to velocity/epsilon, Li & He 2025
   "Back to Basics"; v-prediction for DDPM); dropout; weight decay.
4. Anything else from recent papers (use web search) that fits the rules and promises a big gain.

**Stage 4 — pick the final** from the evidence collected (no extra seeds unless the reviewer asks).

**Stage 5 — final (only when the reviewer says so)**: `final/lab06_final.py` = the winner, cleaned up, with
at the top: a brief abstract (≤ 8 comment lines, no KID claims needed), references (real papers with year
and venue/arXiv id), and an **accurate** AI acknowledgement (Claude Code by Anthropic designed, implemented and
ran the experiments and drafted the abstract; Claude reviewed results and planned the stages; Hamzah
directed and checked the work). Diff the fixed sections against the starter, then run it **once**, from
`~/p6/final`, as `python lab06_final.py` so it writes `lab-06-xmzj54.zip`. Report the zip path, KID and hashes.

## 7. Reference papers you may build on

Ho et al. 2020 (DDPM); Nichol & Dhariwal 2021 (Improved DDPM, cosine schedule); Dhariwal & Nichol 2021 (ADM);
Song et al. 2021 (score SDE / DDPM++); Karras et al. 2022 (EDM); Karras et al. 2024 (EDM2, training
dynamics, power-function EMA); Lipman et al. 2023 (Flow Matching); Liu et al. 2023 (Rectified Flow);
Peebles & Xie 2023 (DiT, adaLN-Zero); Perez et al. 2018 (FiLM); Liu et al. 2022 (ConvNeXt);
Li & He 2025 (x-prediction, "Back to Basics"); Karras et al. 2024 (autoguidance — grey area here).
Verify every reference you cite.
