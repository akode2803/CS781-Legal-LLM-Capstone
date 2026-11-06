# CS781 capstone: Phases 3 and 4 (Indian legal NLP)

All training runs on **Colab**. Nothing in this folder needs a GPU; the local scripts are CPU checks and offline analysis.

## Layout

```
common/                    course scaffold: legal_scaffold/, training_methods.py, tokenizer/ (the Phase 1/2 bundle contents)
                           download_data.py   fetch a phase's files from the gated HF dataset (token from the environment)
phase3_rr/                 rhetorical roles (finished)
  rr_lib.py  doclib.py     libraries that go to Drive   (rr_lib.py is v3)
  notebooks/               phase3_1 ... phase3_9, ensemble, two-stage; Colab notebooks
  analysis/                CPU-only offline analysis (TF-IDF baseline, data facts, document-head CV)
phase4_cjpe/               court judgment prediction (binary outcome)
  cjpe_lib.py              library that goes to Drive
  notebooks/               phase4_0 ... phase4_5, ensemble; Colab notebooks
  make_notebooks.py        regenerates the notebooks (edit defaults here, not in the .ipynb)
  tools/                   local checks: synthetic_smoke.py, notebook_dry_run.py, real_smoke.py, real_baseline.py, dups.py, inspect_data.py
```

## Google Drive layout (what the notebooks expect, under `MyDrive/legal-llm-course/`)

```
phase2_lner_multimethod_bundle.zip     legal_scaffold + training_methods + tokenizer (already there from Phase 2)
rr_lib.py                              from phase3_rr/        (v3)
cjpe_lib.py                            from phase4_cjpe/      (copy it here whenever it changes)
Phase_4_CJPE/Train/{train,dev}.json    downloaded automatically on first run (needs HF_TOKEN in Colab secrets)
Phase_4_CJPE/Test/test.json
runs/                                  Phase 1 snapshots (read) and runs/phase4_cjpe/<preset>/<method>/<run_id>/ (written)
```

## Phase 4 workflow (Colab)

1. Put `cjpe_lib.py` and `rr_lib.py` on Drive as above. In each Colab session, add the `HF_TOKEN` secret (read-only token) and select a GPU runtime (CPU is fine for notebooks 0 and ensemble).
2. **`phase4_0_data_and_baseline`**  (CPU, minutes). Prints case lengths and label balance, then a TF-IDF + logistic-regression baseline on the whole case and on each third. The language-model runs must beat it (about 0.63 to 0.65 dev macro-F1 on a subsample of train).
3. **Run notebooks 1, 2, 3 in parallel**, each in its own Colab session:
   - `phase4_1_mha_pre_full`  MHA-prenorm, full fine-tuning (also the distillation teacher)
   - `phase4_2_gqa_lorma`     GQA-prenorm, LoRMA (compulsory technique)
   - `phase4_3_mla_lora`      MLA-prenorm, LoRA
4. **`phase4_4_gqa_kd_student`** after notebook 1 has finished (it loads `.../MHA-prenorm/full/mha-full-001/best.pt` as the frozen teacher): 6-layer GQA student, cross-entropy + teacher KL.
5. `phase4_5_mha_post_partial`  optional 4th Phase 1 architecture (last two blocks + head). Phase 3 showed this model is the least stable.
6. Each run's cell 6 writes **two submissions**: `submission/` (threshold 0.5) and `submission_tuned/` (threshold that maximises dev macro-F1; chosen on the dev set, so slightly optimistic). Upload the file `submission.zip` itself, not the folder. Cell 6 also saves `dev_probs.json` / `test_probs.json` for the ensemble.
6c. **`phase4_7_mha_pre_lorma`, `phase4_8_mla_lorma`** (L4, about 10 to 15 minutes each, my estimate): architecture controls. Same method (LoRMA), settings and seed as notebook 2; only the architecture differs, so GQA vs MHA vs MLA can be compared without the method confound (run 2 = GQA, run 1 = MHA full, run 3 = MLA LoRA).
6d. **`phase4_9_gqa_lora`, `phase4_10_gqa_full`**: method controls on GQA (same architecture, settings and seed as notebook 2; only LoRA / full fine-tuning instead of LoRMA). With 7/8 this completes an architecture x method comparison.
6b. **`phase4_6_frozen_probe`** (L4, about 10 to 15 minutes; built by `make_probe_notebook.py`): no fine-tuning. Frozen Phase 1 features + logistic regression vs TF-IDF on the same cases, plus a random-weight control. Answers whether the representations are the bottleneck.
7. **`phase4_ensemble`** (CPU): averages the runs' probabilities, tunes the threshold on dev, writes one more submission.
8. Cell 7 of each run notebook is the analysis for the report (per-class scores, false positives/negatives by length, which sections matter via an eval-time chunk-selection ablation, attention by position, cost per case). Cell 8 compares every finished run.

Knobs live in cell 1 of each notebook: `SELECT` (first | last | uniform | head_tail | random), `AGG` (mean | attn | bigru), `MAX_CHUNKS`, `CHUNK_LEN`, `DOCS_PER_EPOCH`, `EPOCHS`, `MAX_MINUTES`. To change a default for all notebooks, edit `phase4_cjpe/make_notebooks.py` and run it.

## Finished Phase 4 submissions (local copies, validated)

`phase4_cjpe/submissions/` holds `submission.zip` copies downloaded from Drive, named `<run>_<model>_<method>__<thr0.5|tuned>.zip`
(runs 1, 2, 3, 4 and 5, plus `ens_*` ensembles built by `tools/local_ensemble.py`). Each has only `prediction.json` at the root and all 1,517 test ids in
`test.json` order. The 0.5 and tuned versions of one run are alternatives, not two experiments: upload one per run.

## Facts about the Phase 4 data (measured)

- train 37,387 rows -> 31,677 after removing duplicate-text rows; dev 1,988 -> 994 (every dev case appears twice); test 1,517. No overlap between splits.
- train is 41% label 1; dev is 50/50, so the 0.5 threshold is biased toward label 0 on dev (the dev-tuned threshold corrects for that). Nothing is known or assumed here about the test label balance: test labels are private.
- cases average about 5,500 Phase 1 tokens (median about 4,200, p90 about 10,000): 11 chunks of 512 on average; 8 chunks fully cover about 48% of cases.
- TF-IDF thirds: last third 0.629 > middle 0.595 > first 0.576 dev macro-F1, so the runs default to `SELECT = 'last'`.
- `test.json` contains a `label` field for every case even though test labels are private. **The loader discards it and nothing here uses it.**

## Local checks (CPU, optional)

```bash
python phase4_cjpe/tools/synthetic_smoke.py          # every architecture x method on a tiny random model with synthetic cases
python phase4_cjpe/tools/notebook_dry_run.py         # runs all Phase 4 notebooks' cells with Drive/Colab mocked
# with real data downloaded:
set -a; . /path/to/HF_TOKEN.env; set +a              # file holds HF_TOKEN=...; never commit it
python common/download_data.py ./data phase4
python -I phase4_cjpe/tools/real_smoke.py ./data     # real files, tiny model, full test export
python -I phase4_cjpe/tools/real_baseline.py ./data  # lengths, balance, TF-IDF-by-section baseline
```

`.gitignore` excludes `*.env`, data folders, checkpoints and arrays. These scripts need `torch`, `tokenizers`, `scikit-learn`, `huggingface_hub`, `nbformat`.

## Phase 5: legal statute identification (multi-label, 100 statutes)

Layout: `phase5_lsi/lsi_lib.py` (library that goes to Drive), `phase5_lsi/make_notebooks.py` (regenerates `phase5_lsi/notebooks/`), `phase5_lsi/tools/` (synthetic smoke test, notebook dry run, token-cache builder).
Drive additions under `MyDrive/legal-llm-course/`: `lsi_lib.py`, the NEW `cjpe_lib.py` (it gained a `pool` option and a collate fix; Phase 4 notebooks still work with it),
`Phase_5_LSI/tokens_lsi_v1_max40000.npz` (build with `phase5_lsi/tools/build_token_cache.py`, about 114 MB; or the first run builds it) and the three Phase 5 data files (downloaded automatically with the `HF_TOKEN` secret).

Notebooks, same pattern as Phase 4: `phase5_0_data_and_baseline` (CPU), runs 1 to 5 (full fine-tune MHA-pre = distillation teacher, LoRMA GQA, LoRA MLA, distilled 6-layer GQA student, partial fine-tune MHA-post), `phase5_ensemble` (CPU).
Each run writes `submission_global/` (one dev-tuned threshold) and `submission_perlabel/` (per-statute thresholds), both with at least one statute per document, plus `dev_probs.npy` / `test_probs.npy`.

Measured facts: train 42,750 rows (42,513 after dropping exact duplicates), dev 10,181 (10,155), test 13,019 (no labels); 100 statutes, all present in train and dev, 272 to 11,731 training documents each (not rare, but 43x imbalanced);
3.7 statutes per document on average (max 31); documents average 1,909 tokens (median 768, p90 4,710), 8 chunks of 512 cover 87% of them. Several rows share a text but have different label sets (generic snippets), so only exact duplicates are removed.
Baseline without a language model (TF-IDF + one-vs-rest LinearSVC, 12,000 training documents, dev subset of 3,000): macro-F1 0.340 with one tuned global threshold, 0.361 with per-statute thresholds scored by 2-fold cross-fitting on dev; micro-F1 0.46.

## Large pretraining run (in progress): corpus preparation

Corpus (Hugging Face `Exploration-Lab/CS-781-Capstone`, `Phase_1_Pretraining/Train/`): `train_acts.json` 0.73 GB, `train_supreme_court.json` 1.32 GB, `train_tribunals.json` 6.48 GB,
`train_high_courts.json` 10.67 GB (19.2 GB total). Records carry metadata (court, year, word and character counts) plus `text`.
`common/download_data.py <dir> phase1` downloads it (measured about 8.5 MB/s from this Mac, so roughly 40 minutes).
`pretrain/prepare_corpus.py` streams it into uint16 token shards (resumable, about 1 GB of RAM); `pretrain/tests/test_prepare_corpus.py` checks the cleaning counts, chunk-invariant reading and interrupted == uninterrupted runs.
Measured on real slices on an M2 (8 GB RAM): 2.2M tokens/s end to end, 3.7 to 4.0 characters per token, so the full corpus is estimated at about 4.3 to 4.6B tokens (about 9 GB of shards) and 35 to 45 minutes of processing.
The previous Phase 1 models saw about 125M legal tokens (batch 2 x 1,024 positions x 60,762 steps), about 2.7% of the corpus.

## Rules to keep in mind

- Every experiment starts from your own Phase 1 model; no other pretrained weights, no external LLM/API.
- Submission format: `submission.zip` containing only `prediction.json` at its root; Phase 4 is a list of `{"id": ..., "pred": 0|1}`.
- Individual competition: do not share code or models. Declare LLM use as the course requires.
