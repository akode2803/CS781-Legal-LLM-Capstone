# CS781 capstone: Indian legal NLP with self-trained language models

Aryan Kumar (261110012), CS781 Large Language Models, IIT Kanpur, 2026.

Seven Codabench phases on Indian court documents (dataset `Exploration-Lab/CS-781-Capstone` on Hugging Face, gated). Every model in every phase starts
from **our own Phase 1 pretraining**; no outside pretrained weights, embeddings or LLM APIs are used in any pipeline. The final report
(`report/latex/report.tex`, PDF alongside) explains the methods and results. This README describes the code, how to run it, and every submission.

All training ran on Google Colab. This repository holds code only: model weights, checkpoints and datasets are kept out (`.gitignore`).

## Results

| Phase | Task | Metric | Best test score | Rank (Nov 8) |
|---|---|---|---|---|
| 1 | Pretraining + legal text continuation (Task 1) | BERTScore F1 | 0.8298 | 84 / 142 |
| 2 | Legal named-entity recognition | strict macro-F1 | 0.5443 | 52 / 132 |
| 3 | Rhetorical role labelling | macro-F1 | 0.4642 | 89 / 131 |
| 4 | Court judgment prediction | macro-F1 | 0.5992 | 122 / 130 |
| 5 | Legal statute identification | macro-F1 | 0.3965 | 41 / 129 |
| 6 | Judgment summarization | BERTScore F1 | 0.8450 | 42 / 124 |
| 7 | Prior case retrieval | Micro-F1@7 | 0.4938 | 8 / 121 |

## Repository layout

```
common/            course scaffold used by Phases 2 to 5: legal_scaffold/ (MHA, GQA, MLA decoders; RoPE, SwiGLU, RMSNorm), training_methods.py
                   (full, partial, LoRA, LoRMA, IA3, distillation), the frozen course BPE tokenizer (32,768 tokens), download_data.py
phase1_original/   the first four Phase 1 models (Oct 4 to 6) and their helper library
pretrain/          clean-slate pretraining on the full corpus (Oct 27 to 28): GQA decoder, MLM encoder, Hymba-style hybrid
phase2_lner/       Phase 2, legal NER
phase3_rr/         Phase 3, rhetorical roles
phase4_cjpe/       Phase 4, judgment prediction
phase5_lsi/        Phase 5, statute identification
phase6_summ/       Phase 6, summarization
phase7_pcr/        Phase 7, prior case retrieval
report/            LaTeX report, figures, training logs, the full experiment record (EXPERIMENTS_FULL.md) and the Codabench history
make_submission.py builds the course submission zip (code/phaseN/ + report PDF) from this repository
```

From Phase 3 on, each phase folder follows the same pattern: a library (`*_lib.py`) with data loading, model, training and export code; a
`make_notebooks.py` that generates the Colab notebooks in `notebooks/` (edit defaults there, not in the `.ipynb`); `tools/` with CPU checks (synthetic
smoke tests, notebook dry runs) and offline analysis. `FINDINGS.md` files hold the detailed experiment notes of Phases 6 and 7.
Phases 4 to 7 keep their submission zips in `submissions/`.

## Setup

- Python 3.10+, `torch`, `tokenizers`, `numpy`, `scipy`, `scikit-learn`, `huggingface_hub`, `nbformat`; Phase 6 validation also uses `rouge-score` and `bert-score`.
- Data: request access to `Exploration-Lab/CS-781-Capstone`, put a read token in a file outside the repo (`HF_TOKEN=...`), then
  ```bash
  set -a; . /path/to/HF_TOKEN.env; set +a
  python common/download_data.py ./data phase4      # phase1 ... phase7
  ```
  In Colab, add `HF_TOKEN` as a notebook secret instead; the notebooks download their data on first run.
- Colab notebooks read and write `MyDrive/legal-llm-course/`: the phase library (`rr_lib.py`, `cjpe_lib.py`, `lsi_lib.py`, `summ_lib.py`, `pcr_lib.py`,
  `rerank_lib.py`), `phase2_lner_multimethod_bundle.zip` (= `common/` plus `phase2_lner/legal_ner.py`), the Phase 1 checkpoints under `runs/` and `pretrain/`,
  and per-run outputs under `runs/<phase>/`. Local scripts that read that folder find it through Google Drive for Desktop, or through `COURSE_DRIVE`.

## Phase 1: pretraining and Task 1

**First models** (`phase1_original/`, Oct 4 to 6). Four causal decoders of about 63 to 67M parameters from the course scaffold: MHA pre-norm,
GQA pre-norm, MLA pre-norm and MHA post-norm, all with RoPE, SwiGLU and RMSNorm and the frozen course tokenizer. Training: one WikiText pass, then legal
text in 100M, 200M and 300M token stages with each of the four sources (acts, Supreme Court, tribunals, High Courts) sampled at 25%. Batch 2 x 1,024 tokens,
60,762 steps, about 125M legal tokens (about 2.7% of the corpus). Run path: `legal_course_end_to_end_v1.ipynb` with `helpers/` (see `RUN_INSTRUCTIONS.md`).
These models are the backbones of Phases 2 to 5.

**Clean-slate pretraining** (`pretrain/`, Oct 27 to 28). Phases 2 to 5 showed the first models were weak, so three ~214M models were trained from scratch
on 4.0B tokens (about 20 tokens per parameter):
- `prepare_corpus.py` streams the 19.2 GB corpus into uint16 token shards (drops empty, short and exact-duplicate documents; resumable).
- `audit_overlap.py` removes 5,470 corpus documents that overlap Phase 4 and 5 dev/test texts (sampled 16-gram fingerprints).
- `train.py`: AdamW (0.9, 0.95; weight decay 0.1 on matrices), lr 6e-4, 300 warm-up steps, cosine to 6e-5, batch 524,288 tokens, 7,629 steps, bf16,
  `torch.compile`, deterministic data order for exact resume. Notebooks: `notebooks/pretrain_dec_gqa.ipynb`, `pretrain_enc_mlm.ipynb`, `pretrain_hybrid.ipynb`.
- Models: GQA decoder (16 layers, d 1024, 16 heads / 4 KV heads, context 2,048; final val loss 1.236), MLM encoder (same layout, bidirectional, 30% masking,
  `encoder.py`), and a Hymba-style hybrid (`hybrid.py`: attention heads and Mamba-2 SSD heads read the same input in parallel; val loss 1.189).
  An MLA decoder was dropped after `gpu_benchmark.py` showed it would cost about twice as much compute.
- Tests: `tests/test_prepare_corpus.py`, `test_hybrid.py` (SSD scan against the recurrence), `test_downstream_loading.py`.
- Task 1 generation: `generate_task1.py` (greedy, optional no-repeat n-gram, repetition penalty, nucleus; ROUGE proxy on held-out prompt tails).

| Submission | BERTScore F1 |
|---|---|
| GQA pre-norm, greedy | 0.7991 |
| MHA pre-norm, greedy | 0.7980 |
| MLA pre-norm, greedy | 0.7950 |
| MHA post-norm, greedy | 0.7369 |
| nucleus (p 0.9, T 0.8) | 0.8211 |
| greedy, no repeated token 4-gram | 0.8228 |
| **greedy, no repeated token 4-gram (second export)** | **0.8298** |

All Phase 1 submissions used the first models (Oct 6). The cell that applied the 4-gram rule was not saved separately; `generate_task1.py --no-repeat 4`
implements the same rule for the new models, whose continuations were analysed but not submitted.

## Phase 2: legal named-entity recognition (`phase2_lner/`)

Word-level BIO tagging with 25 tags over 12 entity types. The backbone reads 512-token windows (stride 384); each word takes the state of its last
BPE piece; a BiLSTM (hidden 256) and a CRF with illegal transitions masked decode the tags (Viterbi). Character offsets come from the tokenizer's
per-piece spans. Settings: AdamW, backbone lr 1e-5, head 1e-3, 100 warm-up updates then cosine, clip 1.0, dropout 0.2, up to 20 epochs with patience 4,
15% of training documents held out. `legal_ner.py` also has a span classifier (spans up to 32 tokens), built but not submitted.

Files: `bio_crf.py` (model and training), `phase2_legal_ner_bio_bilstm_crf.ipynb` (GQA), `phase2_legal_ner_mha_bio_bilstm_crf.ipynb` (MHA),
`phase2_legal_ner_gqa_mha_bio_ensemble.ipynb` + `bio_ensemble.py` (fixed-weight combination of the two taggers' CRF scores),
`bio_best_test_export_v4.py`, `phase2_legal_ner_multimethod.ipynb` (method comparison).

| Submission | Strict macro-F1 |
|---|---|
| MHA pre-norm, best epoch 1 | 0.4647 |
| GQA pre-norm, best epoch 6 | 0.5388 |
| GQA, latest epoch 15 | 0.5417 |
| 6-layer distilled GQA student | 0.5054 |
| GQA LoRMA, epoch 3 | 0.0155 (probably a broken export) |
| **GQA epoch 6 + MHA epoch 1, 50/50 ensemble** | **0.5443** |

## Phase 3: rhetorical role labelling (`phase3_rr/`)

Each sentence gets one of 13 roles. The backbone encodes every sentence (up to 63 tokens) with three neighbouring sentences on each side; a BiGRU runs over
windows of 12 sentences; square-root class weights. Library `rr_lib.py` (+ `doclib.py`); notebooks `phase3_1` to `phase3_9` (one run each: MHA full,
GQA LoRMA, MLA LoRA, MHA IA3, distillation, a sentence-only ablation, distillation onto LoRMA, position features), `phase3_ensemble`, `phase3_twostage`.
`analysis/` holds CPU baselines (TF-IDF with neighbours and position: dev 0.396) and data checks.

| Submission | Run | Macro-F1 |
|---|---|---|
| submission_1 | MLA + LoRA | 0.4509 |
| submission_2 | GQA + LoRMA or MHA + IA3 | 0.4543 |
| **submission_3** | **GQA + LoRMA or MHA + IA3** | **0.4642** |
| submission_4 | MHA pre-norm, full | 0.4508 |
| submission | sentence-only ablation | 0.3532 |
| submission_5 to 7 | distillation onto LoRMA, MLA LoRA + position, MLA full + position (order not recorded) | 0.4342, 0.4386, 0.4294 |

Codabench names were generic, so the mapping of submissions 2 and 3 to the GQA LoRMA and MHA IA3 runs was not recorded.

## Phase 4: court judgment prediction (`phase4_cjpe/`)

Binary outcome from the full case text. Cases (about 5,500 tokens on average) are split into 512-token chunks; up to 8 are selected (`last` by default,
chosen from a TF-IDF analysis of which third of a case predicts best), encoded, pooled by attention into one case vector, and classified with
square-root class weights. Library `cjpe_lib.py`; notebooks from `make_notebooks.py`:

- `phase4_0_data_and_baseline` (CPU): lengths, label balance, TF-IDF baseline (dev 0.624, 0.654 with a tuned threshold).
- Runs: `phase4_1_mha_pre_full` (also the distillation teacher), `phase4_2_gqa_lorma`, `phase4_3_mla_lora`, `phase4_4_gqa_kd_student` (6-layer student),
  `phase4_5_mha_post_partial`; controls `phase4_7` to `phase4_10` vary only the architecture or only the method; `phase4_6_frozen_probe` tests frozen features.
- Each run writes a threshold-0.5 and a dev-tuned-threshold submission plus `dev_probs.json` / `test_probs.json`; `tools/local_ensemble.py` and
  `phase4_ensemble.ipynb` average runs. `ensemble/probs/` keeps the probabilities used for the final ensemble.
- `tools/`: synthetic smoke test of every architecture x method, notebook dry run with Colab mocked, real-data smoke test, duplicate checks.

| Submission | Macro-F1 (thr 0.5 / dev-tuned) |
|---|---|
| run 1, MHA pre-norm full | 0.5805 / 0.5774 |
| run 2, GQA LoRMA | 0.5854 / Failed (0.5862) |
| run 3, MLA LoRA | 0.5665 / 0.5703 |
| run 4, GQA distilled student | 0.5846 / 0.5717 |
| run 5, MHA post-norm partial | - / 0.4877 |
| run 7, MHA LoRMA (control) | 0.5854 / 0.5741 |
| run 8, MLA LoRMA (control) | 0.5761 / - |
| run 9, GQA LoRA (control) | 0.5648 / - |
| run 10, GQA full (control) | 0.5838 / 0.5824 |
| runs 1 + 2, picked on dev | 0.5805 / 0.5912 |
| **runs 1 to 4, equal average** | 0.5912 / **0.5992** |

## Phase 5: legal statute identification (`phase5_lsi/`)

Multi-label prediction over 100 statutes. Same chunked encoder as Phase 4 (uniform chunk selection, attention pooling), 100 sigmoid outputs, BCE with
square-root positive weights, at least one statute per document. Thresholds: one global, per statute, or per statute shrunk toward the global one
(thr_k = λ t_k + (1 − λ) g, λ = n_k / (n_k + m)), always scored by cross-fitting on dev. Library `lsi_lib.py`; notebooks: `phase5_0` (data and
TF-IDF + LinearSVC baseline), runs `phase5_1` to `phase5_5` (first models), `phase5_6` to `phase5_11` (new GQA, encoder, hybrid; two seeds each),
`phase5_tfidf_variants`, `phase5_ensemble` (neural average blended with TF-IDF, thresholds tuned by cross-fit). `tools/run_notebook_local.py` runs the CPU notebooks locally.

| Submission | Macro-F1 (global / per-label) |
|---|---|
| run 1, MHA pre-norm full | 0.3008 / 0.3129 |
| run 2, GQA LoRMA | 0.2667 / 0.2774 |
| run 3, MLA LoRA | 0.2800 / 0.2883 |
| run 4, GQA distilled student | 0.2616 / 0.2818 |
| run 5, MHA post-norm partial | 0.0846 / 0.1019 |
| ensembles (Oct 28) | 0.3843, 0.3765 |
| **tuned ensemble, most likely TF-IDF weight 0.5, m = 30** | **0.3965** |
| TF-IDF weight 0.9 + neural, m = 30 | 0.4180 (scored, but marked Failed by Codabench) |

## Phase 6: summarization (`phase6_summ/`)

The decoder writes the summary after `[document] <eos> " Summary:"`, with the loss on summary tokens only; the encoder runs as a prefix LM.
Library `summ_lib.py` (input selection, training for full / LoRA / LoRMA / distillation, decoding with no-repeat n-grams, beam search and a length policy,
ROUGE and BERTScore validation). Notebooks: `phase6_1` to `phase6_6` (GQA full, LoRMA, LoRA, distilled 8-block student, hybrid full, encoder full),
`phase6_7` (position-weighted loss), `phase6_8_hybrid_doc1536` (final), `phase6_8b_db1536_test_shard` + `test_shard.py` + `merge_shards.py`
(test decoding split across sessions), `make_lead.py` (lead baseline). Details: `FINDINGS.md`.

| Submission | BERTScore F1 |
|---|---|
| lead baseline (first k words) | 0.8256 |
| GQA full / LoRA / LoRMA / distilled student | 0.8350 / 0.8303 / 0.8330 / 0.8299 |
| encoder full (prefix LM) | 0.8345 |
| hybrid full | 0.8356 |
| hybrid, no-repeat 6 (two length settings) | 0.8329, 0.8319 |
| hybrid, length 0.18, no-repeat 3, 4 beams / 6 beams | 0.8409 / 0.8409 |
| **hybrid, document budget 1,536 tokens, 4 beams** | **0.8450** |

## Phase 7: prior case retrieval (`phase7_pcr/`)

For each query judgment, return the 7 cited prior cases from a candidate pool. Pipeline:
1. Lexical shortlist (`pcr_lib.py`): hashed word 4-gram overlap with length normalisation and a header-date filter (gold is always older). Dev 0.4603.
2. Learned ranker (`ltr_features.py`, `ltr_probe.py`): HistGradientBoosting on 19 overlap, length and date features over the top 100. Dev 0.4847.
3. Citing-context features (`ltr_ctx.py`): overlap with the query sentences around each `<CITATION>` placeholder. Dev 0.4947.
4. Cross-encoder rerankers on the new Phase 1 models (`rerank_lib.py`; notebooks `phase7_rerank`, `phase7_cite`, `phase7_cite_lorma`, `phase7_seeds`,
   `phase7_crossfit`, `phase7_controls`, generated by `make_notebook.py`): [query excerpt] <eos> [candidate excerpt] <eos>, softmax over one gold and seven
   lexical negatives. Full, LoRA, LoRMA, IA3, partial and distillation were compared.
5. Final blend (`ltr_ctx_submit.py`): ranker log-odds + 0.2 x mean z-score of three rerankers. Dev 0.4992.

Analysis: `report_analysis.py`, `report_controls.py`, `report_controls_blend.py`, `report_figures.py`, `report/EVIDENCE.md`, `FINDINGS.md`.
`citation_residue.py` measures citation traces left by the organisers' masking; it is analysis only and was never used for a submission.
`chunk_scores.py` / `chunk_eval.py` (window scoring) gave no dev gain and were not submitted.

| Submission | Micro-F1@7 |
|---|---|
| word 4-gram overlap | 0.4554 |
| learned ranker, lexical features | 0.4790 |
| ranker + reranker: encoder full / GQA LoRMA / GQA KD / GQA full | 0.4849 / 0.4867 / 0.4849 / 0.4838 |
| ranker and rerankers retrained on train + dev (five entries) | 0.4784 to 0.4867 |
| citing-context ranker alone | 0.4891 |
| + encoder / + LoRMA / + 4-epoch encoder | 0.4926 / 0.4914 / 0.4908 |
| **+ mean of three rerankers** | **0.4938** |
| + mean of eight rerankers | 0.4932 |
| variable-length lists (`vark_*.py`) | Failed: the rules require exactly 7 per query |

## Building the course submission

```bash
python make_submission.py
```

writes `dist/CS781-CP-Aryan-Kumar-261110012-report_code.zip` with `code/phase1/` to `code/phase7/` and `report/CS781-CP-Aryan-Kumar-261110012-report.pdf`
at the root. It refuses to build if it finds weights, credentials, files over 5 MB, or a report PDF older than its source.

## LLM use

Declared in full in the report's LLM usage declaration: Claude (from Phase 3 on, and for the report) and ChatGPT (earlier phases).
