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

## Running a phase notebook (Phases 3 to 7)

1. Generate the notebooks if you changed a default: `python phase4_cjpe/make_notebooks.py` (likewise `phase5_lsi`, `phase6_summ`, `pretrain`;
   Phase 7 uses `phase7_pcr/make_notebook.py`). Phase 3 notebooks are edited directly.
2. Copy the phase library to Drive: `rr_lib.py`, `cjpe_lib.py`, `lsi_lib.py` and `summ_lib.py` go in `MyDrive/legal-llm-course/`; the Phase 7 files
   (`pcr_lib.py`, `rerank_lib.py`, `rr_lib.py`, `pretrain/encoder.py`, `pretrain/hybrid.py`) go in `MyDrive/legal-llm-course/phase7_pcr/`.
   `phase2_lner_multimethod_bundle.zip` and the Phase 1 checkpoints must already be there (see Setup).
3. Open the notebook in Colab, add the `HF_TOKEN` secret, pick a GPU, and run all cells. Cell 1 holds every setting; `ACTION = 'Train'` trains,
   evaluates on dev and writes the submission; `ACTION = 'Predict'` reloads `best.pt` and only rewrites the submission.
4. The run folder `MyDrive/legal-llm-course/runs/<phase>/<arch>/<method>/<run id>/` receives the log, `best.pt`, dev metrics and `submission.zip`
   (`prediction.json` at the root, the Codabench format). In Phases 3 to 6 training saves its state every epoch (Phase 6: every 20 minutes), so a
   disconnected runtime resumes where it stopped when the notebook is run again.

Shared fine-tuning methods (`common/training_methods.py`): `full` (all weights), `partial` (last blocks only), `lora` and `rslora` (rank-r updates on
query and value), `lorma` (multiplicative low-rank update), `ia3` (learned scaling of keys, values and FFN), `kd` (a student made of every other
block of the backbone, trained on the labels and on a finished `full` teacher's soft outputs, temperature 2).

## Phase 1: pretraining and Task 1

**Architecture.** First models (`phase1_original/`, Oct 4 to 6): four causal decoders of about 63 to 67M parameters from the course scaffold,
12 layers, d_model 512, 8 heads, FFN 2,048, context 2,048, tied embeddings, RoPE, SwiGLU, RMSNorm, the frozen course BPE tokenizer (32,768 tokens):
MHA pre-norm, GQA pre-norm (2 KV heads), MLA pre-norm and MHA post-norm. Clean-slate models (`pretrain/`, Oct 27 to 28), about 214M each:
a GQA decoder (16 layers, d 1,024, 16 heads / 4 KV heads, context 2,048), an MLM encoder with the same layout but bidirectional attention
(`encoder.py`), and a Hymba-style hybrid (`hybrid.py`) whose attention heads and Mamba-2 SSD heads read the same input in parallel.

**Method.** Next-token prediction. First models: one WikiText pass, then legal text in 100M, 200M and 300M token stages, each of the four sources
(acts, Supreme Court, tribunals, High Courts) sampled at 25%. Phases 2 to 5 showed these models were weak, so the clean-slate models were trained on
4.0B tokens (about 20 tokens per parameter) of the deduplicated corpus, with 30% masking for the encoder. Task 1 continues a 512-word excerpt by
128 words.

**Implementation.**
- `phase1_original/legal_course_end_to_end_v1.ipynb` with `helpers/` (scaffold, data preparation, staged training, export); `RUN_INSTRUCTIONS.md` gives every step.
- `pretrain/prepare_corpus.py` streams the 19.2 GB corpus into uint16 token shards (drops empty, short and exact-duplicate documents; resumable).
- `pretrain/audit_overlap.py` removes 5,470 corpus documents that overlap Phase 4 and 5 dev/test texts (sampled 16-gram fingerprints).
- `pretrain/train.py`: one resumable pass with a deterministic data order; `gpu_benchmark.py` measured cost per architecture (MLA was dropped: about twice the compute).
- `pretrain/generate_task1.py`: greedy, optional no-repeat n-gram, repetition penalty, nucleus; ROUGE proxy on held-out prompt tails.
- Tests: `pretrain/tests/test_prepare_corpus.py`, `test_hybrid.py` (SSD scan against the recurrence), `test_downstream_loading.py`.

**Experimental setup.** First models: batch 2 x 1,024 tokens, 60,762 steps, about 125M legal tokens (about 2.7% of the corpus). Clean slate: AdamW
(0.9, 0.95; weight decay 0.1 on matrices), lr 6e-4, 300 warm-up steps, cosine to 6e-5, batch 524,288 tokens, 7,629 steps, bf16, `torch.compile`.
Final validation loss: GQA 1.236, hybrid 1.189.

**How to run.**
```bash
python pretrain/prepare_corpus.py --raw /path/to/Phase_1_Pretraining/Train --out /path/to/tokens
python pretrain/train.py --tokens /path/to/tokens --scaffold /path/to/scaffold --out /path/to/out --tokens-budget 4e9 --micro-batch 32 --compile
python pretrain/generate_task1.py gqa /path/to/model_final.pt --out out_gqa.json
```
In Colab, `pretrain/notebooks/pretrain_dec_gqa.ipynb`, `pretrain_enc_mlm.ipynb` and `pretrain_hybrid.ipynb` run the same steps (they embed `train.py`).

**Submissions** (all from the first models, Oct 6):

| Submission | What it was | BERTScore F1 |
|---|---|---|
| GQA pre-norm | greedy continuation | 0.7991 |
| MHA pre-norm | greedy | 0.7980 |
| MLA pre-norm | greedy | 0.7950 |
| MHA post-norm | greedy | 0.7369 |
| nucleus | nucleus sampling, p 0.9, temperature 0.8 | 0.8211 |
| no-repeat 4 | greedy, no repeated token 4-gram | 0.8228 |
| **no-repeat 4, second export** | **same rule, re-exported** | **0.8298** |

The cell that applied the 4-gram rule was not saved separately; `generate_task1.py --no-repeat 4` implements the same rule for the new models,
whose continuations were analysed but not submitted.

## Phase 2: legal named-entity recognition (`phase2_lner/`)

**Architecture.** A first-model backbone (GQA or MHA pre-norm) reads the text; each word takes the hidden state of its last BPE piece; a BiLSTM
(hidden 256) and a CRF over 25 BIO tags (12 entity types) with illegal transitions masked.

**Method.** Word-level BIO tagging, decoded with Viterbi; character offsets come from the tokenizer's per-piece spans. The best submission adds the
CRF scores of the GQA and MHA taggers with fixed 50/50 weights. `legal_ner.py` also has a span classifier (spans up to 32 tokens), built but not submitted.

**Implementation.** `bio_crf.py` (alignment, model, training, recovery), `bio_ensemble.py` (score ensemble), `bio_best_test_export_v4.py`
(test export), `phase2_legal_ner_gqa_mha_bio_ensemble_recovery.py`.

**Experimental setup.** 512-token windows with stride 384; AdamW, backbone lr 1e-5, head lr 1e-3, 100 warm-up updates then cosine, gradient clip 1.0,
dropout 0.2, up to 20 epochs with patience 4, 15% of training documents held out for model selection. Metric: strict macro-F1.

**How to run.** In Colab, run `phase2_legal_ner_bio_bilstm_crf.ipynb` (GQA) and `phase2_legal_ner_mha_bio_bilstm_crf.ipynb` (MHA), then
`phase2_legal_ner_gqa_mha_bio_ensemble.ipynb` to combine them and write the submission. `phase2_legal_ner_multimethod.ipynb` compares methods.

**Submissions:**

| Submission | What it was | Strict macro-F1 |
|---|---|---|
| MHA | MHA pre-norm tagger, best epoch 1 | 0.4647 |
| GQA | GQA pre-norm tagger, best epoch 6 | 0.5388 |
| GQA latest | GQA, epoch 15 | 0.5417 |
| student | 6-layer distilled GQA student | 0.5054 |
| LoRMA | GQA with LoRMA, epoch 3 | 0.0155 (probably a broken export) |
| **ensemble** | **GQA epoch 6 + MHA epoch 1, 50/50 CRF scores** | **0.5443** |

## Phase 3: rhetorical role labelling (`phase3_rr/`)

**Architecture.** A first-model backbone encodes each sentence (up to 63 tokens) together with three neighbouring sentences on each side; a BiGRU runs
over windows of 12 sentence vectors; a linear layer predicts one of 13 roles.

**Method.** Sentence classification in context, square-root class weights. Runs vary the backbone and the fine-tuning method; one ablation drops the
neighbours (sentence only), two add sentence-position features.

**Implementation.** `rr_lib.py` (data, windows, model, training, export) with `doclib.py`; notebooks `phase3_1` to `phase3_9` (one run each),
`phase3_ensemble`, `phase3_twostage`. `analysis/` holds CPU baselines (TF-IDF with neighbours and position: dev 0.396) and data checks.

**Experimental setup.** 8 epochs, seed 17; learning rates (backbone / adapter / head) full 5e-5 / - / 1e-3, LoRA and LoRMA - / 1e-3 / 1e-3,
IA3 - / 3e-3 / 1e-3; LoRA rank 8, alpha 16 on query and value; distillation temperature 2, alpha 0.5. Metric: macro-F1.

**How to run.** Copy `rr_lib.py` to `MyDrive/legal-llm-course/`, then run a notebook such as `notebooks/phase3_2_gqa_lorma.ipynb`.
`phase3_5_gqa_kd_needs_run1.ipynb` needs the finished run 1 as its teacher.

**Submissions:**

| Submission | What it was | Macro-F1 |
|---|---|---|
| submission_1 | MLA + LoRA | 0.4509 |
| submission_2 | GQA + LoRMA or MHA + IA3 | 0.4543 |
| **submission_3** | **GQA + LoRMA or MHA + IA3** | **0.4642** |
| submission_4 | MHA pre-norm, full | 0.4508 |
| submission | sentence-only ablation | 0.3532 |
| submission_5 to 7 | distillation onto LoRMA, MLA LoRA + position, MLA full + position (order not recorded) | 0.4342, 0.4386, 0.4294 |

Codabench names were generic, so the mapping of submissions 2 and 3 to the GQA LoRMA and MHA IA3 runs was not recorded.

## Phase 4: court judgment prediction (`phase4_cjpe/`)

**Architecture.** Cases average about 5,500 tokens. Each case is cut into 512-token chunks; up to 8 are selected and encoded by a first-model backbone;
attention pooling turns the chunk vectors into one case vector; a linear layer predicts the binary outcome.

**Method.** Chunk selection `last` (a TF-IDF analysis showed the last third of a case predicts best), square-root class weights. Each run also tunes a
decision threshold on dev. The best submission averages the probabilities of runs 1 to 4 with equal weights and a dev-tuned threshold.

**Implementation.** `cjpe_lib.py` (data, chunking, model, training, export); notebooks from `make_notebooks.py`:
- `phase4_0_data_and_baseline` (CPU): lengths, label balance, TF-IDF baseline (dev 0.624, 0.654 with a tuned threshold).
- Runs `phase4_1_mha_pre_full` (also the distillation teacher), `phase4_2_gqa_lorma`, `phase4_3_mla_lora`, `phase4_4_gqa_kd_student` (6-layer student),
  `phase4_5_mha_post_partial`; controls `phase4_7` to `phase4_10` change only the architecture or only the method; `phase4_6_frozen_probe` tests frozen features.
- `tools/local_ensemble.py` and `phase4_ensemble.ipynb` average runs; `ensemble/probs/` keeps the probabilities of the final ensemble.
- `tools/`: synthetic smoke test of every architecture x method, a notebook dry run with Colab mocked, a real-data smoke test, duplicate checks.

**Experimental setup.** 5 epochs of 2,000 random cases each, at most 60 minutes per run (T4), batch 4 cases x 2 accumulation steps, AdamW (0.9, 0.98),
6% warm-up then cosine; learning rates (backbone / adapter / head) full 3e-5 / - / 5e-4, partial 1e-4 / - / 5e-4, adapters - / 1e-3 / 5e-4,
IA3 - / 3e-3 / 5e-4; rank 8, alpha 16; distillation temperature 2, alpha 0.3. Metric: macro-F1.

**How to run.** Copy `rr_lib.py` and `cjpe_lib.py` to Drive, run a notebook in `notebooks/`, and after runs 1 to 4:
```bash
python -I phase4_cjpe/tools/local_ensemble.py /path/to/data phase4_cjpe/ensemble/probs out/ runs=1,2,3,4
```

**Submissions:**

| Submission | What it was | Macro-F1 (thr 0.5 / dev-tuned) |
|---|---|---|
| run 1 | MHA pre-norm, full | 0.5805 / 0.5774 |
| run 2 | GQA, LoRMA | 0.5854 / Failed (0.5862) |
| run 3 | MLA, LoRA | 0.5665 / 0.5703 |
| run 4 | GQA distilled 6-layer student | 0.5846 / 0.5717 |
| run 5 | MHA post-norm, partial | - / 0.4877 |
| run 7 | MHA, LoRMA (control) | 0.5854 / 0.5741 |
| run 8 | MLA, LoRMA (control) | 0.5761 / - |
| run 9 | GQA, LoRA (control) | 0.5648 / - |
| run 10 | GQA, full (control) | 0.5838 / 0.5824 |
| runs 1 + 2 | pair picked on dev | 0.5805 / 0.5912 |
| **runs 1 to 4** | **equal-weight average** | 0.5912 / **0.5992** |

## Phase 5: legal statute identification (`phase5_lsi/`)

**Architecture.** The Phase 4 chunked encoder (uniform chunk selection, attention pooling) with 100 sigmoid outputs, one per statute. Later runs use
the clean-slate GQA decoder, encoder and hybrid as backbones.

**Method.** Multi-label BCE with square-root positive weights, at least one statute per document. Thresholds: one global, one per statute, or per
statute shrunk toward the global one (thr_k = λ t_k + (1 − λ) g, λ = n_k / (n_k + m)), always scored by cross-fitting on dev. The best submission
averages six neural runs, blends them with a TF-IDF model, and uses the shrunk thresholds.

**Implementation.** `lsi_lib.py`; notebooks `phase5_0` (data and TF-IDF baseline), runs `phase5_1` to `phase5_5` (first models), `phase5_6` to
`phase5_11` (clean-slate GQA, encoder and hybrid, two seeds each), `phase5_tfidf_variants`, `phase5_ensemble`. `tools/run_notebook_local.py` runs the
CPU notebooks locally.

**Experimental setup.** 4 epochs of 12,000 random documents each (of about 42,000), at most 60 minutes per run, batch 4 x 4 accumulation steps,
8 chunks of 512 tokens, rank 8, alpha 16; per-epoch evaluation on 3,000 dev documents. Metric: macro-F1.

**How to run.** Copy `rr_lib.py`, `cjpe_lib.py` and `lsi_lib.py` to Drive, run the run notebooks, then `phase5_tfidf_variants` and `phase5_ensemble`
(CPU; locally with `python phase5_lsi/tools/run_notebook_local.py`).

**Submissions:**

| Submission | What it was | Macro-F1 (global / per-label threshold) |
|---|---|---|
| run 1 | MHA pre-norm, full | 0.3008 / 0.3129 |
| run 2 | GQA, LoRMA | 0.2667 / 0.2774 |
| run 3 | MLA, LoRA | 0.2800 / 0.2883 |
| run 4 | GQA distilled student | 0.2616 / 0.2818 |
| run 5 | MHA post-norm, partial | 0.0846 / 0.1019 |
| ensembles (Oct 28) | neural averages with TF-IDF | 0.3843, 0.3765 |
| **tuned ensemble** | **six neural runs + TF-IDF, most likely TF-IDF weight 0.5, shrinkage m = 30** | **0.3965** |
| TF-IDF 0.9 | TF-IDF weight 0.9 + neural, m = 30 | 0.4180 (scored, but marked Failed by Codabench) |

## Phase 6: summarization (`phase6_summ/`)

**Architecture.** The clean-slate decoders write the summary after `[document] <eos> " Summary:"`; the encoder runs as a prefix LM. The final model
is the Hymba-style hybrid (attention and Mamba-2 heads in parallel, about 214M parameters).

**Method.** Full fine-tuning with the loss on summary tokens only. The document is cut to a token budget (first 3/4 and last 1/4 of the budget).
Decoding: beam search, no repeated token 3-gram, summary length = clip(0.18 x document words, 200, 700) words, within 0.8 to 1.2 of that.

**Implementation.** `summ_lib.py` (input selection, training for full / LoRA / LoRMA / distillation, decoding with no-repeat n-grams, beam search and
the length policy, ROUGE and BERTScore validation). Notebooks `phase6_1` to `phase6_6` (GQA full, LoRMA, LoRA, distilled 8-block student, hybrid full,
encoder full), `phase6_7` (position-weighted loss), `phase6_8_hybrid_doc1536` (final), `phase6_8b_db1536_test_shard` + `test_shard.py` + `merge_shards.py`
(test decoding split across sessions), `make_lead.py` (lead baseline). Every experiment is written up in `FINDINGS.md`.

**Experimental setup.** 2 epochs, batch 2 x 8 accumulation steps (16 sequences), context 2,048, AdamW (0.9, 0.95), 5% warm-up then cosine, lr 3e-5
(full) or 5e-4 (LoRA, LoRMA), rank 16, alpha 32. Final run: document budget 1,536 tokens, 4 beams. Metric: BERTScore F1 (ROUGE also tracked).

**How to run.** Copy `summ_lib.py` to Drive, run `notebooks/phase6_8_hybrid_doc1536.ipynb`; to split test decoding, run
`phase6_8b_db1536_test_shard.ipynb` once per shard, then
```bash
python phase6_summ/merge_shards.py <run dir> <reference submission.zip> submission.zip
python phase6_summ/make_lead.py /path/to/Phase_6_Summ lead.zip        # lead baseline
```

**Submissions:**

| Submission | What it was | BERTScore F1 |
|---|---|---|
| lead | first k words of the document, no model | 0.8256 |
| GQA | full / LoRA / LoRMA / distilled student | 0.8350 / 0.8303 / 0.8330 / 0.8299 |
| encoder | full fine-tune as a prefix LM | 0.8345 |
| hybrid | full fine-tune | 0.8356 |
| hybrid, no-repeat 6 | two length settings | 0.8329, 0.8319 |
| hybrid, tuned decoding | length 0.18, no-repeat 3, 4 beams / 6 beams | 0.8409 / 0.8409 |
| **hybrid, budget 1,536** | **document budget 1,536 tokens, 4 beams** | **0.8450** |

## Phase 7: prior case retrieval (`phase7_pcr/`)

**Architecture.** A three-stage ranker: a lexical shortlist, a gradient-boosted learned ranker, and cross-encoder rerankers built on the clean-slate
models (the pair `[query excerpt] <eos> [candidate excerpt] <eos>` goes through the backbone, a linear head scores it).

**Method.** For each query judgment, return exactly 7 cited prior cases from the candidate pool.
1. Lexical shortlist (`pcr_lib.py`): hashed word 4-gram overlap with length normalisation and a header-date filter (gold cases are always older). Dev 0.4603.
2. Learned ranker (`ltr_features.py`, `ltr_probe.py`): HistGradientBoosting on 19 overlap, length and date features over the top 100. Dev 0.4847.
3. Citing-context features (`ltr_ctx.py`): overlap with the query sentences around each `<CITATION>` placeholder. Dev 0.4947.
4. Rerankers (`rerank_lib.py`): softmax over one gold and seven lexical negatives; full, LoRA, LoRMA, IA3, partial and distillation compared.
5. Final blend (`ltr_ctx_submit.py`): ranker log-odds + 0.2 x mean z-score of three rerankers. Dev 0.4992.

**Implementation.** Lexical and ranker stages run on CPU; rerankers run in notebooks generated by `make_notebook.py` (`phase7_rerank`, `phase7_cite`,
`phase7_cite_lorma`, `phase7_seeds`, `phase7_crossfit`, `phase7_controls`). Analysis: `report_analysis.py`, `report_controls.py`,
`report_controls_blend.py`, `report_figures.py`, `report/EVIDENCE.md`, `FINDINGS.md`. `citation_residue.py` measures citation traces left by the
organisers' masking; it is analysis only and was never used for a submission. `chunk_scores.py` / `chunk_eval.py` (window scoring) gave no dev gain.

**Experimental setup.** Rerankers: 512 query + 512 candidate tokens, 7 negatives from the lexical top 40 for training, top 50 reranked at test time,
2 epochs, accumulation 4, AdamW (0.9, 0.95), lr 2e-5 (full) or 3e-4 (LoRA, LoRMA), head lr 1e-4, rank 16, alpha 32, at most 40 minutes per run.
Metric: Micro-F1@7.

**How to run.**
```bash
python phase7_pcr/ltr_features.py feats.pkl                       # ranker features for train, dev, test (reads the course Drive folder)
python phase7_pcr/ltr_ctx.py feats.pkl ctx.pkl                    # add citing-context features
python phase7_pcr/ltr_ctx_submit.py ctx.pkl ranker                # ranker alone
python phase7_pcr/ltr_ctx_submit.py ctx.pkl blend=<run>,<run>,<run>:0.2   # ranker + mean z-score of rerankers
```
The reranker runs come from the Colab notebooks above (`RUNS` in `make_notebook.py` lists every run).

**Submissions:**

| Submission | What it was | Micro-F1@7 |
|---|---|---|
| lexical | word 4-gram overlap | 0.4554 |
| ranker | learned ranker, lexical features | 0.4790 |
| ranker + one reranker | encoder full / GQA LoRMA / GQA KD / GQA full | 0.4849 / 0.4867 / 0.4849 / 0.4838 |
| train + dev | ranker and rerankers retrained on train + dev (five entries) | 0.4784 to 0.4867 |
| context ranker | citing-context ranker alone | 0.4891 |
| context ranker + one reranker | encoder / LoRMA / 4-epoch encoder | 0.4926 / 0.4914 / 0.4908 |
| **context ranker + three rerankers** | **mean z-score of three rerankers** | **0.4938** |
| context ranker + eight rerankers | mean of eight | 0.4932 |
| variable length | `vark_*.py` lists of varying length | Failed: the rules require exactly 7 per query |

## Building the course submission

```bash
python make_submission.py
```

writes `dist/CS781-CP-Aryan-Kumar-261110012-report_code.zip` with `code/phase1/` to `code/phase7/` and `report/CS781-CP-Aryan-Kumar-261110012-report.pdf`
at the root. It refuses to build if it finds weights, credentials, files over 5 MB, or a report PDF older than its source.

## LLM use

Declared in full in the report's LLM usage declaration: Claude (from Phase 3 on, and for the report) and ChatGPT (earlier phases).
