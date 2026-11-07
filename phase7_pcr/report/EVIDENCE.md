# Phase 7 report evidence

Facts and numbers only, laid out in the order of the course template. The report text built from these facts is `phase7_sections.tex`. Dev numbers come from `report/analysis.json` (`report_analysis.py`), run logs and `FINDINGS.md`.

## Section 1: competition result

Best valid: 1040042, Micro-F1@7 0.4938 (P 0.5039, R 0.4841). Rank 5 at 16:30 IST on Nov 7; re-read the final rank after the 23:59 IST close. 17 submissions used; 7+ are different methods.

## Section 3: data

| Split | Queries | Candidates | Gold mean | Gold median | Gold max |
|---|---|---|---|---|---|
| Train | 827 | 4320 | 6.4 | 4 | 52 |
| Dev | 118 | 1023 | 8.2 | 5 | 92 |
| Test | 237 | 1727 | not public | | |

- Dev query length: median 4451 words, p90 14684. Dev candidates: median 3869, p90 16082.
- Every dev query is also in the dev pool (118 of 118), so self is removed. 920 of 1023 dev pool documents (89.9%) are gold for at least one dev query.
- Citations, names and organisations are masked as `<CITATION>`, `<NAME>`, `<ORG>`. About 12.7 `<CITATION>` placeholders per dev query. Spearman between placeholders and gold count per query: 0.939.
- Temporal: gold is older than the query in 352 of 352 pairs where both `DATE OF JUDGMENT` headers parse; the header parses for about half of the documents.
- Perfect @7 ranking on dev: F1 0.618 (P 0.673, R 0.571, 556 hits).
- Text was not scrubbed: leftover citation traces remain in it, and no feature matches them on purpose. Deliberate matching was measured on dev and train, rejected, and never submitted.

## Section 4: system as built

1. Lexical stage: hashed word 4-gram overlap, max_df .05, length norm b .6, self removed, later candidates dropped when both header dates parse. Recall on dev: @7 .425, @20 .623, @50 .742, @100 .791.
2. Learned ranker (not a Phase 1 model): HistGradientBoosting, 150 iterations, lr .05, 15 leaves, 5 seeds averaged, fit on train, 27 columns: overlap at 3, 4, 5, 6, 8-grams (log and per-query z), pool-level mean overlap, rank, lengths, date gap, and 8 citing-context columns (n = 3, 4; window 1, 2; log and z).
3. Neural stage (Phase 1 models): cross-encoder over the lexical top 50. Input: 512-token query excerpt `<eos>` 512-token candidate excerpt `<eos>`. Excerpt = 96-token opening, then sentences sharing the most rare 4-grams with the other document; `cite=True` fills the query side from sentences within 1 of a `<CITATION>` first. Linear head on last-token (GQA) or mean (encoder) pooling.
4. Loss: softmax cross-entropy over 1 gold + 7 negatives sampled from the lexical top 40, at most 3 gold per query per epoch (6 in the 4-epoch runs).
5. Final score: z(ranker logit) + 0.2 x mean z(neural of p7-enc-full, p7c-enc, p7c-lorma); candidates with no neural score get -2.

Failed or unhelpful: neural score as ranker input; train + dev retraining; 4-epoch training; per-citation-site features; averaging seeds. Numbers in `FINDINGS.md`.

## Section 5: settings

| Setting | Value |
|---|---|
| GPU | A100 on Colab, bf16 autocast |
| Optimizer | AdamW, betas .9/.95, weight decay .01, clip 1.0 |
| Schedule | 5% warmup, cosine, floor 5% |
| Batch | 1 group of 8 pairs, accumulation 4 |
| Epochs | 2 (4 for p7c-enc, p7c-lorma) |
| LR | full 2e-5; KD and partial 3e-5; LoRMA and LoRA 3e-4; IA3 3e-3; head 1e-4 |
| Adapters | rank 16, alpha 32, on query and value |
| KD | T 2.0, alpha .5, CE + KL to p7-gqa-full; student keeps blocks 0, 2, ..., 14 |
| Partial | top 4 blocks (12-15) |
| Epoch selection | blended dev F1; neural-only dev F1 for the p7c runs |

No GPU run tested pooling, loss or negative sampling one factor at a time, and there is no frozen or random-init control. Those template prompts have no direct evidence yet.

## Section 6: techniques (same pipeline, dev)

Neural F1 and AUC are inside the lexical top 50. Blend columns: lexical blend w .25, and citing-context ranker + 0.2 z(neural) (ranker alone .4947, 445 hits).

| Run | Params (M) | Trainable | Min | ms/pair | Peak GB | Neural F1 | AUC | Lexical blend | Ranker blend |
|---|---|---|---|---|---|---|---|---|---|
| Encoder full | 213.9 | 100% | 11.7 | 4.5 | 12.0 | .3391 | .771 | .4725 | .4969 |
| GQA full (teacher) | 213.9 | 100% | 9.8 | 4.2 | 12.0 | .3168 | .756 | .4703 | .4914 |
| GQA LoRMA | 214.6 | 0.31% | 8.5 | 4.5 | 6.2 | .3335 | .784 | .4658 | .4914 |
| GQA KD student | 123.8 | 100% | 7.9 | 2.1 | 7.4 | .2079 | .646 | .4658 | .4958 |
| GQA IA3 | 214.0 | 0.03% | 8.9 | 4.7 | 6.8 | .3002 | .738 | .4647 | .4936 |
| GQA partial | 213.9 | 21.08% | 5.2 | 4.2 | 3.7 | .3168 | .762 | .4625 | .4958 |
| GQA LoRA | 214.8 | 0.40% | 8.8 | 4.7 | 6.4 | .3135 | .760 | .4603 | .4958 |

AUC inside the top 50 for reference: lexical .915, citing-context ranker .950.

Paired bootstrap over dev queries, neural-only F1 against GQA full (5000 resamples, 95% interval): encoder +.0222 [.0032, .0422]; LoRMA +.0167 [-.0031, .0364]; partial .0000 [-.0179, .0179]; LoRA -.0033 [-.0233, .0169]; IA3 -.0167 [-.0394, .0062]; KD -.1089 [-.1385, -.0806].

Rank correlation inside the top 50: GQA full with encoder .69, LoRMA .76, KD .82, partial .80; LoRMA with LoRA .88, IA3 .80. Each model with lexical order: -.01 to .17.

## Section 6: pipeline steps

| Step | Dev F1 | Dev hits | Test F1 | Paired dev gain, 95% interval |
|---|---|---|---|---|
| Lexical | .4603 | 414 | .4554 | |
| + learned ranker | .4847 | 436 | .4790 | +.0245 [.0095, .0410] |
| + citing-context columns | .4947 | 445 | .4891 | +.0100 [-.0010, .0214] |
| + three-reranker term | .4992 | 449 | .4938 | +.0044 [.0010, .0094] |

`FINDINGS.md` gives the old ranker as .4858 in its first table (`ltr_probe.py`) and .4847 in the citing-context table. This rebuild gives .4847, and the two differ by a few hits. The blend weight and the three-model choice came from about 15 dev combinations, so the last row's dev interval is optimistic. One hit is about .0006 test F1.

## Section 7: errors (dev, final blend)

- Gold: 973. Hit 449. In the final top 20: 642, top 50: 739, top 100: 770. Outside the top 100: 203. Beyond the 7-slot cap: 417 gold cannot be retrieved by any system.
- Precision by rank 1 to 7: .915, .686, .627, .508, .458, .331, .280.
- Queries with no hit: 5 of 118.
- False positives: 377 (313 unique). 94.2% are gold for another dev query; the pool base rate is 89.9%.
- Mean per-query z of 4-gram overlap: hits 3.37, false positives 1.52, gold missed within the top 100 0.56.
- The neural term swapped 58 entries in the top 7: 21 gold in, 17 gold out.
- Recall by how many dev queries cite the gold case: 1: .462, 2: .463, 3-5: .412.

| Gold per query | Queries | Gold | Hits | P | R | F1 |
|---|---|---|---|---|---|---|
| 1-2 | 33 | 51 | 44 | .190 | .863 | .312 |
| 3-5 | 30 | 127 | 89 | .424 | .701 | .528 |
| 6-10 | 27 | 202 | 134 | .709 | .663 | .685 |
| 11-20 | 19 | 259 | 120 | .902 | .463 | .612 |
| 21+ | 9 | 334 | 62 | .984 | .186 | .312 |

| Query words | Queries | Gold | R | | Gold candidate words | Pairs | R |
|---|---|---|---|---|---|---|---|
| <2k | 18 | 37 | .649 | | <1k | 112 | .375 |
| 2-4k | 35 | 166 | .675 | | 1-2k | 122 | .541 |
| 4-8k | 39 | 300 | .573 | | 2-4k | 254 | .543 |
| 8-16k | 15 | 181 | .414 | | 4-8k | 257 | .506 |
| 16k+ | 11 | 289 | .228 | | 8k+ | 228 | .320 |

Query length and gold count move together (Spearman .615), so the query-length column is not a clean length effect. Hits as a share of reachable slots, min(gold, 7), by query length: .649, .742, .823, .872, .904.

Gold candidates in queries with at most 7 gold, by candidate length:

| Words | Pairs | Recall | In lexical top 7 | In lexical top 100 |
|---|---|---|---|---|
| <1k | 40 | .500 | .350 | .675 |
| 1-2k | 44 | .727 | .614 | .841 |
| 2-4k | 78 | .756 | .692 | .872 |
| 4-8k | 73 | .836 | .795 | .890 |
| 8k+ | 48 | .667 | .583 | .812 |

Figures: `fig_gold_size.pdf`, `fig_length.pdf`.
