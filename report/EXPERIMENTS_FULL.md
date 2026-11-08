# CS781 capstone: every experiment, in full

This is the uncompressed record for the final report, built on Nov 8, 2026. It is the source for the 10-page report, not the report itself. Each number says where it comes from: **test** = Codabench score, **dev** = our held-out split, **CV** = cross-fit on dev or train. The raw Codabench history is in `codabench_history.txt` next to this file.

Phase 1 to 3 Codabench scores were added on Nov 8 from the submission names.

## 0. Competition results at a glance

| Phase | Task | Metric | Best test | Leaderboard entry | Rank (read Nov 8) |
|---|---|---|---|---|---|
| 1 | Pretraining + Task 1 generation | BERTScore F1 | 0.8298 (first models, greedy + no repeated token 4-gram) | 965440 | below the 50 rows shown |
| 2 | Legal NER | strict macro-F1 | 0.5443 (GQA + MHA BIO ensemble) | 978750 | below the 50 rows shown |
| 3 | Rhetorical roles | macro-F1 | 0.4642 | 994180 | below the 50 rows shown |
| 4 | Judgment prediction | macro-F1 | 0.5992 | 1009488 (ensemble of runs 1 to 4, tuned threshold) | below the 50 rows shown |
| 5 | Statute identification | macro-F1 | 0.3965 on the board; 0.4180 scored but marked Failed | 1016808 | 41 of 130 rows |
| 6 | Summarization | BERTScore F1 | 0.8450 (ROUGE-1 0.4591, ROUGE-L 0.2459) | 1030690 | 42 of 122 rows (34 of 50 on Nov 3) |
| 7 | Prior case retrieval | Micro-F1@7 | 0.4938 (P 0.5039, R 0.4841) | 1040042 | 6 of 132 rows (Phase 7 closes Nov 8 23:59) |

"Rows" counts every row in the results table, so it overstates the number of teams.

## 1. Models

### 1.1 Original Phase 1 models (used in Phases 2 to 5)

Four ~63 to 67M-parameter causal decoders from the course scaffold (`legal_scaffold`): MHA pre-norm (67.4M), GQA pre-norm (62.7M), MLA pre-norm (67.0M), MHA post-norm (67.4M). All use RoPE, SwiGLU, RMSNorm and the frozen course BPE tokenizer (vocab 32,768, sha 906194848c16…). Training: WikiText stage, then legal text in 100M → 200M → 300M token stages with balanced 25% sampling per source, batch 2 × 1,024 for 60,762 steps, about 125M legal tokens in all (about 2.7% of the corpus). Codabench zips: `legal-llm-course/Codabench submissions - 4 architectures - legal 100M - 2026-10-06/`.

Task 1 test BERTScore (Oct 6): greedy GQA 0.7991, MHA pre 0.7980, MLA 0.7950, MHA post 0.7369; nucleus p 0.9 T 0.8 0.8211; greedy + no repeated token 4-gram 0.8228, then 0.8298 (best). Phase 1 validation losses of these models are not recorded here.

### 1.2 Clean-slate pretraining (Oct 27 to 28)

**Why.** The Phase 2 to 5 results pointed at weak representations: language-model runs barely beat TF-IDF in Phases 3 to 5. The old models had seen only about 2.7% of the corpus. We retrained from scratch at about 20 tokens per parameter (Chinchilla ratio).

**Corpus.** Hugging Face `Exploration-Lab/CS-781-Capstone`, Phase 1 train: acts 0.73 GB, supreme court 1.32 GB, tribunals 6.48 GB, high courts 10.67 GB (19.2 GB). Tokenized on the Mac (`pretrain/prepare_corpus.py`) into uint16 shards: drop empty, short (<200 characters) and exact-duplicate documents; one pass in natural proportions (no balancing); 3.7 to 4.0 characters per token; about 4.3 to 4.6B tokens in all.

**Contamination audit** (`pretrain/audit_overlap.py`). Sampled token 16-gram fingerprints of the Phase 4 and 5 dev and test texts against the corpus. Dropped 5,470 corpus documents sharing 20 or more sampled windows. The Phase 2 and 3 audit was never run.

**Recipe** (`pretrain/train.py`, identical for all three): AdamW (betas 0.9/0.95, wd 0.1 on matrices only), lr 6e-4, warmup 300, cosine to 6e-5, clip 1.0, batch 524,288 tokens, 7,629 steps = 4.0B tokens, bf16, chunked cross-entropy, torch.compile, deterministic data order for exact resume.

| Model | Shape | Params | Objective | Speed (G4) | Cost | Final val loss |
|---|---|---|---|---|---|---|
| GQA decoder | 16 layers, d 1024, 16 heads, 4 KV heads, FFN 2816, context 2048 | 214M | causal LM | 130 to 132k tok/s | ~77 CU, ~8.7 h | 1.236 |
| MLM encoder | same weights layout, bidirectional attention, window 512 | 214M | masked LM, 30% masking | 172k tok/s | | 0.686 (MLM loss, not comparable) |
| Hymba-style hybrid | 16 layers, parallel GQA attention (8 heads) + Mamba-2 SSD heads (8), FFN 2688 | 215M | causal LM | 113k tok/s | ~87 to 90 CU, ~10 h | 1.189 |

- Throughput benchmark (`pretrain/gpu_benchmark.py`, M tokens per CU): L4 47.6, A100 40 GB **59.5**, A100 80 GB 50.1, G4 53.1. An MLA decoder ran at 56k tok/s (177 CU estimated), so it was dropped.
- **GQA vs hybrid, same data order** (val loss at step; corrected Nov 8 from the logs, the earlier version had the labels swapped): 250: 2.675 vs 2.569; 500: 2.042 vs 1.943; 750: 1.758 vs 1.694; 1000: 1.668 vs 1.597; 1250: 1.602 vs 1.539. Logs: `report/logs/`. The hybrid reached GQA's step-1250 loss at about step 1000, about 20% fewer tokens. Final: 1.236 vs 1.189.
- Our hybrid omits Hymba's meta tokens, sliding-window attention and cross-layer KV sharing, so it is "Hymba-style". Its SSD scan is plain PyTorch, checked against the recurrence. It has no KV cache, so generation recomputes the prefix.
- The encoder uses the decoder's code with non-causal attention (`pretrain/encoder.py`), and padding is hidden by an embedding hook.

**Frozen-feature probe** (CJPE, Phase 4 data; frozen features + logistic regression; dev macro-F1 with tuned threshold, best of configurations):

| Features | Dev macro-F1 |
|---|---|
| Random weights | 0.514 |
| Old GQA pre-norm | 0.562 |
| New GQA at 25% | 0.571 |
| New encoder at 25% | 0.587 |
| New encoder final | 0.583 |
| New GQA final | 0.589 |
| Hybrid at 75% | 0.602 |
| TF-IDF, no language model | 0.612 |

Cross-validated F1 on train barely separates the models (hybrid 0.567, GQA 0.567, encoder 0.558), so the ranking among the new models is not established. Pretraining clearly works (random → final +0.07 to +0.09), but frozen features still trail TF-IDF on this task.

### 1.3 Task 1 (generation from the new GQA model)

50 prompts, repeated word-4-gram fraction (rep4): greedy 0.664 with 86% looping outputs; greedy + no-repeat 4-gram 0.019, 0% loops, 92% clean.

ROUGE proxy (60 official prompts: first 384 words as prompt, the next 128 as reference; my own ROUGE, no stemming):

| Decoding | ROUGE-1 | ROUGE-2 | ROUGE-L | rep4 |
|---|---|---|---|---|
| greedy + no-repeat 3 | 0.373 | 0.105 | **0.226** | 0.002 |
| greedy + no-repeat 4 | 0.361 | 0.101 | 0.223 | 0.012 |
| greedy (course default) | 0.290 | 0.076 | 0.216 | 0.625 |
| nucleus T 0.8 | 0.371 | 0.094 | 0.213 | 0.051 |
| nucleus T 0.7 + no-repeat 4 | 0.359 | 0.090 | 0.210 | 0.002 |
| repetition penalty 1.15 | 0.374 | 0.096 | 0.203 | 0.046 |
| copy the previous 128 words | 0.382 | 0.103 | 0.198 | |

The model barely beats copying the prompt's last 128 words. Encoder fill-mask top-1 accuracy at 30% masking: about 0.74 to 0.80. The new GQA model's Task 1 continuations were never submitted (no Phase 1 submissions after Oct 6).

## 2. Phase 2: Legal NER (Oct 11)

**Data** (computed Nov 8 from the HF files). Train 70 judgments in two folds of 35; test 35 (fold 3). Median 4,488 words. 8,393 entities, 120 per document: DATE 18.8%, STAT 15.0%, PREC 14.1%, COURT 12.1%, AUTH 12.0%, CASENO 9.1%, APP 4.8%, RESP 4.5%, JUDGE 2.9%, A.COUNSEL 2.3%, WIT 2.2%, R.COUNSEL 2.2%. Span median 2 words, p95 9, max 35. No overlapping spans; 137 adjacent pairs (only whitespace between). 26 of 3,418 surface strings carry more than one label (e.g. "union of india"). Gold offsets: Unicode code points, end exclusive; 0 text mismatches.

**Model.** Word-level BIO tagging (25 tags): 512-token windows, stride 384; word = backbone state of its last BPE piece; BiLSTM; CRF with illegal transitions masked; Viterbi. Offsets come from the tokenizer's per-piece character spans. A span classifier (`legal_ner.py`, spans up to 32 tokens) was built, not submitted.

| Submission | Test strict macro-F1 |
|---|---|
| MHA pre-norm BIO, best epoch 1 | 0.4647 |
| GQA pre-norm BIO, best epoch 6 | 0.5388 |
| GQA, latest epoch 15 | 0.5417 |
| 6-layer distilled student | 0.5054 |
| GQA LoRMA, epoch 3 | 0.0155 (probably a broken export) |
| GQA epoch 6 + MHA epoch 1, 50/50 ensemble | **0.5443** |

Learning rate, batch and dropout are not recorded here.

## 3. Phase 3: Rhetorical role prediction (Oct 16)

**Task.** Label every sentence of a judgment with one of 13 roles (CL and IT domains). Train 80 documents (16,845 sentences), dev 10 (2,142), test 10. Dev is 10 documents; dev CL has ArgumentPetitioner at 24.6% against 9.2% in train, and no Dissent or PrecedentNotReliedUpon.

**Model.** Old Phase 1 backbones encode each sentence (capped at 63 tokens) with ±3 neighbouring sentences of context; a BiGRU runs over a window of 12 core sentences; sqrt class weights. Variants added sentence-position features (`pos`).

| Run | Backbone + method | Trainable | Dev macro-F1 (present classes) | Test |
|---|---|---|---|---|
| sentence-only | GQA + LoRMA, no context | 0.62% | 0.3414 | 0.3532 |
| 3 | MLA + LoRA, window | 1.10% | 0.4457 | 0.4509 |
| 1 | MHA pre-norm, full | 100% | 0.4622 | 0.4508 |
| 2 | GQA + LoRMA, window | 1.09% | 0.4478 | 0.4543 or 0.4642 (mapping to confirm) |
| 4 | MHA pre-norm + IA3 | 0.89% | 0.4153 | 0.4543 or 0.4642 (mapping to confirm) |
| 7 | GQA + LoRMA + KD (teacher = run 1) | 1.09% | **0.4942** | one of 0.4294 / 0.4342 / 0.4386 |
| 8 | MLA + LoRA + position | 1.11% | 0.4511 | one of 0.4294 / 0.4342 / 0.4386 |
| | MLA full + position | 100% | 0.4602 | one of 0.4294 / 0.4342 / 0.4386 |
| 5 | KD run (phase3_5) | | peaked 0.42 at epoch 2, then dev loss rose from 2.05 to 4.05 while train F1 hit 0.99 | TODO |

Eight submissions scored (submissions 5 to 7 are the three later runs, order not recorded). The ensemble was ready but missed the 23:59 deadline.

**Offline analysis (dev macro-F1).**
- Baselines: always `Fact` 0.031; position + domain 0.172; TF-IDF sentence 0.312; + position + domain 0.335; **+ neighbours ±1 + position (logistic regression) 0.396**; ±2 neighbours 0.386. So the language models add about 5 to 9 points.
- Oracle: copying the gold previous label gives 0.749, so a sequence-aware model has a high ceiling. The paper's 0.70 (Kalamkar et al.) used BERT-class encoders + BiLSTM-CRF over the whole document.
- 17.6% of CL and 22.2% of IT sentences exceed 63 tokens and were truncated.
- A count-based transition prior (Viterbi) hurt: 0.396 → 0.31 to 0.35.
- Document-level sequence models on cheap features: per-sentence MLP 0.319; BiLSTM over the whole document 0.357; + CRF 0.347; + label-shift loss 0.347; + both 0.341.
- Overfitting: MLA LoRA train loss 1.76 → 0.07 while dev F1 peaked at epoch 5. The full fine-tune's dev F1 rose to 0.4622 by epoch 7 while train loss fell to 0.015.
- Dev did not rank runs reliably: MHA IA3 dev 0.4153 vs its test (0.4543 or 0.4642), and the later runs (dev up to 0.4942) scored 0.4294 to 0.4386 on test.

## 4. Phase 4: Court judgment prediction (Oct 21 to 26)

**Data.** Train 37,387 rows → 31,677 after removing duplicate texts; dev 1,988 → 994 (every dev case appears twice); test 1,517. Train is 41% label 1 and dev 50/50. Cases average about 5,500 tokens (median 4,200, p90 10,000).

**Model.** Chunks of 512 tokens (up to 8; selection `last` by default), chunk vectors pooled by attention into a case vector, binary head, sqrt class weights. Each run writes a threshold-0.5 and a dev-tuned-threshold submission.

**Baselines (TF-IDF + logistic regression, dev).** Majority 0.333; first third 0.576; middle third 0.595; last third 0.629; whole case 0.624 (0.654 with a tuned threshold).

| Run | Backbone + method | Trainable | Dev @0.5 | Dev tuned (thr) | Test @0.5 | Test tuned | Train time | Peak GB | ms/case |
|---|---|---|---|---|---|---|---|---|---|
| 1 | MHA pre-norm, full (teacher) | 100% (67.4M) | 0.5776 | 0.6243 (0.42) | 0.5805 | 0.5774 | 7.1 min* | 8.3 | 25.0 |
| 2 | GQA, LoRMA | 0.68% | 0.5966 | 0.6044 (0.49) | 0.5854 | 0.5862 (status Failed) | 5.5 min* | 4.9 | 23.9 |
| 3 | MLA, LoRA | 0.71% | 0.5634 | 0.5649 (0.49) | 0.5665 | 0.5703 | 12.1 min | 5.8 | 30.1 |
| 4 | GQA 6-layer KD student | 100% (39.9M) | 0.5766 | 0.5906 (0.44) | 0.5846 | 0.5717 | 8.2 min | 4.8 | 11.6 |
| 5 | MHA post-norm, partial (last 2 blocks) | 12.9% | 0.4726 | 0.4726 | | 0.4877 | 4.6 min | 1.8 | 25.0 |
| 7 | MHA pre-norm, LoRMA (control) | 0.74% | 0.5866 | 0.5877 | **0.5854** | 0.5741 | 9.9 min | 5.0 | 27.0 |
| 8 | MLA, LoRMA (control) | 0.82% | 0.5553 | 0.5553 | 0.5761 | | 2.3 min | 6.1 | 4.6 |
| 9 | GQA, LoRA (control) | 0.74% | **0.6122** | 0.6122 | 0.5648 | | 3.5 min | 5.0 | 6.3 |
| 10 | GQA, full (control) | 100% | 0.5918 | 0.5995 (0.47) | 0.5838 | 0.5824 | 10.8 min | 8.2 | 22.6 |
| ens | runs 1 to 4, equal average | | 0.5890 | 0.6245 | 0.5912 | **0.5992** | | | |
| ens | runs 1 + 2 (picked on dev) | | 0.5979 | 0.6388 | 0.5805 | 0.5912 | | | |

\* Times from `results.json`, on mixed T4/L4 GPUs, so not comparable across runs.

**What we learned.**
- Dev-tuned thresholds did not transfer: they overstated test by 2.5 points on average (4.7 for run 1) and beat the 0.5 versions by only 0.12 points on test.
- Test SE is about ±1.3 points on 1,517 cases, so runs 1, 2, 4, 7 and 10 are indistinguishable. Run 9 had the best dev and the worst test score of the controls.
- The architecture × method controls (runs 2, 7, 8 for architecture under LoRMA; 2, 9, 10 for method on GQA) show no detectable effect on test.
- The a-priori four-run ensemble (0.5992) beat the dev-selected pair (0.5912); picking subsets on dev did not pay. Fitting the ensemble on dev (nested split): best subset 0.633, stacking 0.618, four-run average 0.622.
- Distillation: the student matched its teacher at 0.5 (0.5766 vs 0.5776) with 41% fewer parameters, 2.2× faster inference and 4.8 vs 8.3 GB in training.
- Every language-model run stayed at or below the TF-IDF baseline on dev.
- Class weighting: sqrt weights move the effective label-1 share from 41.4% to 45.7% (inverse weights 50%).
- Post-norm partial fine-tuning collapsed, as in Phase 3.
- Training curves: runs 3 and 4 were still rising at epoch 5; runs 1 and 2 peaked at epoch 4.

## 5. Phase 5: Legal statute identification (Oct 25 to 28)

**Data.** Train 42,750 rows (42,513 after dropping exact duplicates); dev 10,181 (10,155); test 13,019. 100 statutes, 272 to 11,731 training documents each (43× imbalance), 3.7 statutes per document (max 31). Documents average 1,909 tokens (median 768); 8 chunks of 512 cover 87%.

**Model.** Chunked encoder as in Phase 4 (uniform selection, attention aggregation), 100 sigmoid outputs, BCE with sqrt positive weights. Decision rules: one global threshold, per-statute thresholds, at least one statute per document; per-statute thresholds are scored honestly by 2-fold cross-fit on dev.

**Baseline.** TF-IDF + one-vs-rest LinearSVC on 12,000 documents: macro-F1 0.340 (global), 0.361 (per-label, cross-fit), micro-F1 0.46.

**Old models** (dev global / per-label cross-fit / micro; test global / per-label):

| Run | Backbone + method | Dev | Test | Train time |
|---|---|---|---|---|
| 1 | MHA pre-norm, full | 0.305 / 0.314 / 0.421 | 0.3008 / 0.3129 | 28 min |
| 2 | GQA, LoRMA | 0.268 / 0.278 / 0.386 | 0.2667 / 0.2774 | 23 min |
| 3 | MLA, LoRA | 0.279 / 0.288 / 0.394 | 0.2800 / 0.2883 | 31 min |
| 4 | GQA 6-layer KD student | 0.261 / 0.285 | 0.2616 / 0.2818 | 62 min (stopped at T4 time limit) |
| 5 | MHA post-norm, partial | 0.088 / 0.105 / 0.161 | 0.0846 / 0.1019 | 13 min |

All below the TF-IDF baseline; post-norm failed again.

**New models** (full fine-tune, identical recipe: 4 epochs × 12,000 docs, lr 5e-5 backbone / 1e-3 head, seed 17 and 18):

| Model | Dev global | Dev cross-fit (seed 17) | Cross-fit (seed 18) | Micro | Train time | Peak GB |
|---|---|---|---|---|---|---|
| Encoder | 0.3633 | 0.3703 | 0.3632 | 0.467 | 14.2 min | 20.0 |
| Hybrid | 0.3603 | 0.3667 | 0.3642 | 0.465 | 21.6 min | 28.3 |
| GQA | 0.3539 | 0.3637 | 0.3611 | 0.459 | 13.4 min | 20.0 |
| Old best (MHA, 67M) | 0.3054 | 0.3139 | | 0.421 | | |

Seed noise (0.003 to 0.007) is as large as the gaps, so the three architectures are indistinguishable. All three beat the old best by about 0.05 and the TF-IDF baseline by 0.01 to 0.02. Encoder and hybrid were still gaining about 0.012 from epoch 3 to 4.

**Ensembles (dev cross-fit macro-F1).**

| System | Cross-fit | Test |
|---|---|---|
| Best single run | 0.370 | |
| Six-run neural ensemble, per-label thresholds | 0.3866 | |
| + threshold shrinkage (m = 30) | 0.3853 to 0.3866 | |
| TF-IDF alone (w12_svm0.1) | 0.410 | |
| TF-IDF weight 0.9 + neural, m = 30 | 0.4122 | **0.4180** (1016845, marked Failed) |
| TF-IDF variant w12_svm0.5bal (alone 0.3939), weight 0.5, m = 100 | **0.4203** | not submitted (deadline) |

Threshold shrinkage: thr_k = λ_k t_k + (1 − λ_k) g with λ_k = n_k / (n_k + m), so m = 0 is per-label and m = ∞ is global; chosen by five random 2-fold cross-fit splits.

Earlier Oct 28 submissions: 0.3843 (1016616/1016620), 0.3765 (1016739), 0.3965 (1016808, on the board). TODO: which ensemble each of these was.

**Error analysis.** Rare statutes are the weak spot: mean F1 by train-frequency quartile 0.23 / 0.33 / 0.36 / 0.50. Label co-occurrence stacking was tried (`stack_cooccurrence_cv`).

## 6. Phase 6: Abstractive summarization (Nov 1 to 3)

**Data.** Train 7,030 judgment/headnote pairs, test 100. Document words: median 3,129 (p90 7,855, max 139,645). Summary words: median 638 (p90 1,501). Median summary/document ratio 0.217. Only about 11% of summary sentences appear verbatim in the document. We held out 200 training documents as validation (100 scored).

**Model.** Decoder-as-summarizer: `[document tokens] <eos> " Summary:" [summary] <eos>`, loss on summary tokens only, context 2048 (document budget 1024). Input strategies: head, headtail (first 3/4 + last 1/4), select (lead + tail + TF-IDF-central sentences). The encoder runs as a prefix LM (bidirectional over the document, causal over the summary). Training: 2 epochs, batch 2 × accumulation 8. KD: student keeps 8 of the GQA model's 16 blocks (0, 2, …, 14), teacher = GQA full, T 2.0, α 0.5.

**Methods (validation, greedy, no-repeat 3, ratio 0.22; ROUGE unstemmed, my implementation):**

| Run | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | Val loss (ep1 → ep2) | Train loss | Test BERTScore |
|---|---|---|---|---|---|---|---|
| hybrid full | **0.483** | **0.214** | **0.245** | 550 | 1.7488 → 1.7478 | 1.787 → 1.611 | 0.8356 |
| GQA full | 0.473 | 0.207 | 0.242 | 542 | 1.8201 → 1.8180 | 1.859 → 1.686 | 0.8350 |
| GQA LoRA | 0.465 | 0.199 | 0.238 | 567 | 1.8774 → 1.8697 | 1.916 → 1.867 | 0.8303 |
| GQA LoRMA | 0.464 | 0.199 | 0.235 | 563 | 1.8843 → 1.8776 | 1.921 → 1.877 | 0.8330 |
| encoder full (prefix LM) | 0.460 | 0.201 | 0.235 | 579 | 2.3734 → 2.3028 | 3.205 → 2.238 | 0.8345 |
| GQA KD student (8 blocks) | 0.451 | 0.188 | 0.231 | 557 | 2.2743 → 2.2198 | 2.601 → 2.134 | 0.8299 |
| lead baseline (first k words) | 0.509 | 0.261 | 0.255 | 685 | | | 0.8256 |

**The metric.** BERTScore with roberta-large (layer 17, no idf, no rescaling) truncates both texts at 512 roberta tokens, so only the first ~400 words count. Test references average 932 words (validation 751). Reference summaries repeat 10.5% of their token 3-grams in the first 400 words.

**Decoding and length (hybrid unless stated; validation BERTScore → test):**
- No-repeat size (GQA, validation): nr3 0.8400 > nr4 0.8375 > nr5 0.8361 > nr2 0.8357 (ROUGE-L falls to 0.206) > nr6 0.8346 (13% repeated word 4-grams: paraphrase loops).
- Length: forcing longer output hurt (ratio 0.3, lo 300 to 500, hi 900: 0.8371 to 0.8367 against 0.8378). Shorter helped: ratio 0.18 0.8437, 0.15 0.8443 (greedy, nr3).
- Beam search: 4 beams 0.8469 validation → **0.8409** test; 6 beams 0.8478 → 0.8409 (saturated).
- Context split: document 1,536 tokens, summary ~510 (about BERTScore's window), 4 beams: 0.8492 → **0.8450** test, ROUGE-1 fell (0.459 vs 0.484 on test).
- Position-weighted loss (summary tokens after ~450 words × 0.3): val loss 1.7521 vs 1.7488; no score (compute ran out).
- Validation → test offset: test 0.005 to 0.008 below validation for every submission; gains carried over (+0.0038 validation → +0.0053 test for beams).

Test submissions in order: methods (Nov 2) 0.8299 to 0.8356; nr6 0.8329 and 0.8319; beams 0.8409, 0.8409; context split 0.8450.

## 7. Phase 7: Prior case retrieval (Nov 4 to 8)

Full detail is in `phase7_pcr/FINDINGS.md`, `phase7_pcr/report/EVIDENCE.md`, `ANALYSIS_EXPLAINED.md` and `phase7_sections_full.tex`. The essentials:

**Data.** Train / dev / test: 827 / 118 / 237 queries against pools of 4,320 / 1,023 / 1,727. Dev gold per query: mean 8.2, median 5, max 92; a perfect @7 ranking scores only 0.618. Every dev query is in its own pool. Gold is always older than the query when both header dates parse (352 of 352 pairs). Citations are masked as `<CITATION>`.

**Lexical stage (dev).** BM25 unigrams 0.1012; bigrams 0.2101; trigrams 0.2857. Hashed word n-gram overlap: n3 0.3513, n4 0.3913, n5 0.4191 (b 0.3); final n4, b 0.6, max_df 0.05, header-date filter **0.4603** (test 0.4554). Recall @50 0.742.

**Pipeline and test results.**

| System | Dev F1 | Dev hits | Test F1 | Paired dev gain [95%] |
|---|---|---|---|---|
| Word 4-gram overlap | 0.4603 | 414 | 0.4554 | |
| + learned ranker (HistGradientBoosting, 19 features) | 0.4847 | 436 | 0.4790 | +0.0245 [0.0095, 0.0410] |
| + citing-context features (27 features) | 0.4947 | 445 | 0.4891 | +0.0100 [−0.0010, 0.0214] |
| + three rerankers (final) | 0.4992 | 449 | **0.4938** | +0.0044 [0.0010, 0.0094] |

**Adaptation methods** (cross-encoder on 512 + 512-token excerpts, softmax over 1 gold + 7 negatives from the lexical top 40):

| Method | Params | Trainable | ms/pair | GB | Neural F1 | AUC top 50 | Blend dev | Test |
|---|---|---|---|---|---|---|---|---|
| Encoder full | 213.9M | 100% | 4.5 | 12.0 | 0.3391 | 0.771 | 0.4969 | 0.4849 |
| GQA full (teacher) | 213.9M | 100% | 4.2 | 12.0 | 0.3168 | 0.756 | 0.4914 | 0.4838 |
| GQA LoRMA | 214.6M | 0.31% | 4.5 | 6.2 | 0.3335 | 0.784 | 0.4914 | 0.4867 |
| GQA KD student | 123.8M | 100% | 2.1 | 7.4 | 0.2079 | 0.646 | 0.4958 | 0.4849 |
| GQA IA3 | 214.0M | 0.03% | 4.7 | 6.8 | 0.3002 | 0.738 | 0.4936 | |
| GQA partial (top 4) | 213.9M | 21.08% | 4.2 | 3.7 | 0.3168 | 0.762 | 0.4958 | |
| GQA LoRA | 214.8M | 0.40% | 4.7 | 6.4 | 0.3135 | 0.760 | 0.4958 | |

**One-factor controls** (dev, one seed; seed spread s.d. 0.0097 neural F1): pairwise loss 0.3324 (−0.0067, n.s.); random negatives 0.4002 (+0.0611) but the worst blend (442 vs 447 hits; correlation with the ranker 0.555 vs 0.245); last-token pooling 0.3079 (−0.0311); frozen backbone 0.2379; random start 0.1946 (chance 0.112); GQA mean pooling 0.3357 vs 0.3168 (n.s.).

**Failed ideas.** Neural score as a ranker feature; retraining on train + dev (no test gain: 0.4784 to 0.4867); 4-epoch training (train top-1 0.95, dev neural ~0.32); per-citation-site features; averaging eight rerankers (test 0.4932).

**Error analysis.** 449 of 973 dev gold retrieved; 417 unreachable (queries with more than 7 gold); 203 outside the top 100. Precision 0.915 at rank 1 → 0.280 at rank 7. Long-query recall drop is the 7-slot cap (hits / min(gold, 7) rises 0.65 → 0.90). Missed gold has mean overlap z 0.56 vs 3.37 for hits: citations without shared text.

**Integrity.** Citation traces survive the masking; matching them deliberately raised dev from 0.4603 to 0.4747. You rejected it as cheating; it was never submitted. Text was not scrubbed, and the report must say so. The citing-context features use where `<CITATION>` sits, not which case it is.

## 8. Patterns across phases

1. **Lexical baselines are strong on legal text.** TF-IDF matched or beat the old language models in Phases 3 to 5, was the largest member of the Phase 5 ensemble, and n-gram overlap carries most of Phase 7.
2. **More pretraining helped.** The 214M models trained on 4B tokens beat the old 67M models by about 0.05 macro-F1 in Phase 5, and the probe moved from 0.562 (old) to 0.589 to 0.602.
3. **Architecture differences are within noise** in every downstream phase where we checked seeds or test SE (Phases 4, 5, 7). The hybrid's lower pretraining loss gave a small edge in Phase 6 generation (0.8356 vs 0.8350 test).
4. **Parameter-efficient methods match full fine-tuning** within noise (LoRMA in Phases 4, 6, 7), at about half the memory.
5. **Distillation keeps most quality at ~40% fewer parameters and ~2× speed** (Phases 4, 7), but the student loses standalone quality in Phase 7 (0.2079 vs 0.3168).
6. **Dev selection is fragile.** Tuned thresholds and dev-picked subsets did not transfer in Phase 4; Phase 6 test ran 0.006 below validation.
7. **Post-norm partial fine-tuning failed** in Phases 3 to 5.
8. **Knowing the metric matters.** BERTScore truncation (Phase 6) and the 7-slot cap (Phase 7) explain much of the behaviour.

## 9. Compute (approximate)

Pretraining about 77 + 87 CU for GQA and hybrid, plus the encoder run (TODO: its CU). Phase 7 runs about 8 CU each (A100), controls 12.7 CU. Colab accounts used: aryankumar (Drive), shrijasbps, kumarkunalls.
