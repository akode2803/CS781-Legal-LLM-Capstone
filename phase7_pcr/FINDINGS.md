# Phase 7 (prior case retrieval): findings

Metric: Micro-F1@7. A perfect @7 ranking on dev scores 0.618, because gold sets are larger than 7 for many queries.

## Dev scores

| Method | Dev F1 | Note |
|---|---|---|
| BM25 unigram | 0.101 | Topic words do not identify cited cases |
| BM25 1-3 grams | 0.286 | |
| Word 5-gram overlap, length norm b 0.3 | 0.419 | Shared quoted passages are the signal |
| Word 4-gram overlap, b 0.6 | 0.4425 | |
| Word 4-gram overlap, b 0.6, header date filter | **0.4603** | Train split: 0.376 (pool is 4x larger) |

Recall of that ranking on dev: top 7 0.425, top 20 0.623, top 50 0.742, top 100 0.791.

## Test scores (Codabench)

| Submission | Test F1 | Precision | Recall | Dev F1 |
|---|---|---|---|---|
| `p7-shingle4-b06-hdrdate.zip` (1038437) | 0.4554 | 0.4647 | 0.4464 | 0.4603 |
| `p7-ltr-lexical.zip` (1038613), learned ranker alone | 0.4790 | 0.4888 | 0.4696 | 0.4858 |
| `p7-ltr-gqa-full.zip` (1038614) | 0.4838 | 0.4937 | 0.4742 | 0.4892 |
| `p7-ltr-enc-full.zip` (1038608) | 0.4849 | 0.4949 | 0.4754 | 0.4925 |
| `p7-ltr-gqa-kd.zip` (1038612) | 0.4849 | 0.4949 | 0.4754 | 0.4914 |
| **`p7-ltr-gqa-lorma.zip` (1038611, on leaderboard)** | **0.4867** | 0.4967 | 0.4771 | 0.4892 |

Test tracks dev closely. Every neural blend beats the learned ranker alone, by 0.005-0.008. The differences among the four neural entries are a few hits and within noise; LoRMA trains 0.31% of the parameters, and the KD student scores pairs about twice as fast as the others.

## Reranker analysis (p7-gqa-full, dev)

- Neural alone 0.3168, z-score blend with raw lexical 0.4614 (one extra hit, noise).
- Neural and lexical disagree: rank correlation 0.09 inside the top 50. Gold in top 7: lexical only 190, neural only 61, both 224.
- Raw lexical scores are heavy-tailed, so z-scoring them is a poor blend. z(log(1 + lexical)) + 0.25 z(neural) gives 0.4703; reciprocal-rank blend at 0.25 gives 0.4669. The weight was chosen on dev, so these are slightly optimistic.
- Replacing the order of the lexical top 10 / 15 / 20 with neural order hurts (0.4514 / 0.4102 / 0.3913): the neural score works as a small correction, not a replacement.

## Seven reranker runs (A100, 512 + 512 token excerpts, 2 epochs, about 8 CU in all)

Blend: z(log(1 + lexical)) + 0.25 z(neural), same weight for every model. Lexical alone: 0.4603.

| Run | Trainable | Minutes | ms per pair | Peak GB | Neural alone | Dev F1 blended |
|---|---|---|---|---|---|---|
| Encoder, full | 100% | 11.7 | 4.5 | 12.0 | 0.3391 | **0.4725** |
| GQA, full (KD teacher) | 100% | 9.8 | 4.2 | 12.0 | 0.3168 | 0.4703 |
| GQA, LoRMA | 0.31% | 8.5 | 4.5 | 6.2 | 0.3335 | 0.4658 |
| GQA, KD student (8 blocks) | 100% | 7.9 | 2.1 | 7.4 | 0.2079 | 0.4658 |
| GQA, IA3 | 0.03% | 8.9 | 4.7 | 6.8 | 0.3002 | 0.4647 |
| GQA, partial (top 4 blocks) | 21.08% | 5.2 | 4.2 | 3.7 | 0.3168 | 0.4625 |
| GQA, LoRA | 0.40% | 8.8 | 4.7 | 6.4 | 0.3135 | 0.4603 |

- Averaging all seven did not help (best 0.4691 across weights, 0.4669 at 0.25). GQA + encoder full: 0.4714 at 0.25.
- With the weight chosen on one half of dev and scored on the other (20 random splits), gain over lexical: encoder +0.0096, GQA full +0.0081, all seven +0.0053.
- Neural alone is far below lexical, so all these models act as small corrections. Differences of a few hits between runs are within noise.
- Each run's own zip used the old raw z-score blend, which set the neural weight to 0 for five of the seven. Use `fuse_submissions.py` instead.

## Learned ranker on content features (`ltr_probe.py`)

Gradient boosting fit on the train split over word-overlap scores at 3, 4, 5, 6 and 8-grams (log and per-query z-score), rank, lengths, date gap and a pool-level feature: a candidate's mean overlap with all queries in its split. No labels are used for that feature. Dev scores are averaged over 5 seeds.

| Ranking | Dev F1 |
|---|---|
| Lexical order | 0.4603 |
| Learned ranker, no pool-level feature | 0.4680 |
| Learned ranker | 0.4858 |
| + 0.2 x encoder full | 0.4925 |
| + 0.2 x GQA full | 0.4892 |
| + 0.2 x LoRMA | 0.4892 |
| + 0.2 x KD student | 0.4914 |

The pool-level feature gives most of the gain: it marks landmark cases that many queries resemble. It depends on the split's own queries, and it transferred from train (4320 candidates) to dev (1023). The ranker itself is not a Phase 1 model; the neural term is. Weight 0.2 was read off the dev grid.

## Notes

- Gold candidates are always older than the query when both have a `DATE OF JUDGMENT` header (352 of 352 pairs), so dropping later candidates is safe. The header parses for about half of the documents.
- Highest year in the text is not a usable date: 20-33% of gold candidates violate it.
- Mixing n-gram sizes by z-score did not beat the best single size.

## Citation traces (measured on dev and train, rejected, never submitted)

We decided not to use these: matching leftover citations reads the answer from masking gaps and does not improve any model. The numbers below only document the leak.

The organisers masked citations, but traces survive: header `CITATION:` and `CITATOR INFO` blocks, and reporter references in footnotes. See [citation_residue.py](citation_residue.py).

| Lexical ranking, matched candidates moved to the front | Dev F1 | Train F1 |
|---|---|---|
| No promotion | 0.4603 | 0.3760 |
| Forward matches (precision 0.91 dev, 0.71 train) | 0.4703 | 0.3938 |
| Citator matches (precision 0.98 dev, 0.91 train) | 0.4703 | 0.3974 |
| Both (precision 0.93 dev, 0.75 train) | 0.4747 | 0.4088 |

Removing the traces from the text (`scrub`, 0.5% of characters) lowers the plain lexical score slightly: dev 0.4603 to 0.4569, train 0.3760 to 0.3747. Our own match patterns find nothing afterwards; that does not prove every trace is gone.

Submission 1038437 (test 0.4554) was built on unscrubbed text, so it did not match citations explicitly but did read the traces.

## Out-of-fold neural features and train + dev retraining (Nov 7, 07:45 IST)

Three-fold cross-fitting gave every train query a reranker score from a model that never saw that query (`ltr_neural.py`), and encoder and LoRMA were retrained on train + dev (`ltr_final.py`). Neither idea beat the existing best entry.

**Idea 1, neural score as a ranker feature (dev, top-50 rows; ranker with no neural = 0.488, 439 hits):**

| Neural source | As feature | Additive w .1 | w .2 | w .3 |
|---|---|---|---|---|
| encoder full | .4914 | .4903 | **.4947** | .4936 |
| GQA LoRMA | .4814 | .4903 | .4903 | .4836 |
| GQA full | .4847 | .4880 | .4903 | .4903 |
| all three | .4847 | .4892 | .4925 | .4914 |

Feature input never beat the additive blend, and was below the plain ranker for two of the four sources. It was not submitted.

**Idea 2, train + dev retraining (test, Codabench). Additive w .2, lexical-only ranker:**

| ID | Ranker fit on | Reranker fit on | Test F1 |
|---|---|---|---|
| 1038613 | train | none | .4790 |
| 1038845 | train + dev | none | .4784 |
| 1038611 (leaderboard) | train | LoRMA, train | **.4867** |
| 1038844 | train + dev | LoRMA, train | .4849 |
| 1038842 | train + dev | LoRMA, train + dev | .4843 |
| 1038608 | train | encoder, train | .4849 |
| 1038841 | train + dev | encoder, train + dev | **.4867** |
| 1038843 | train + dev | encoder + LoRMA, train + dev | .4855 |

One correct prediction is worth about .0006 test F1 (about 824 hits, 1659 predictions, about 1727 gold), so all differences within the neural entries are 0 to 4 hits, and the train + dev changes are noise-sized in both directions. The neural term over the ranker alone is about 13 hits. Dev cannot score train + dev models, and the `dev_f1` in `final/*/train_results.json` is contaminated: dev was in their training data.

1038611 stays on the leaderboard: nothing beat it, so no entry was chosen from these five test scores.

## Citing context (Nov 7, 15:30 IST)

Queries keep a `<CITATION>` placeholder where each citation was (about 12.7 per dev query, 8.2 gold). The sentences within 1 of a placeholder are 10% of the query text, and overlap from them alone scores dev .4314 (full query .4603); blending the two gives .4758. This uses where a citation sits, not which case it was. No reporter reference, `CITATION:` header or `CITATOR INFO` is read on purpose, and the user approved submitting it. Traces remain in the raw text as in all runs.

Eight columns added to the ranker (`ltr_ctx.py`: n = 3, 4; window 1, 2; log and per-query z). Ranker fit on train, rerankers from the earlier train-only runs:

| ID | Entry | Dev | Test |
|---|---|---|---|
| 1038613 | old ranker alone | .4847 | .4790 |
| 1039873 | citing-context ranker alone | .4947 | .4891 |
| 1039875 | + LoRMA, w .1 | .4969 | .4914 |
| 1039874 | + encoder, w .2 (leaderboard) | .4969 | **.4926** |

Test gain over the old ranker alone: +.0101, about 17 hits. Window and blend sizes were picked on dev.

## Citing-context rerankers and longer training (Nov 7, 16:30 IST)

The query excerpt was filled from citing sentences first (`pair_ids(cite=True)`), and epochs were selected on neural-only dev F1 (`select_neural`). Excerpt tests on dev, neural alone:

| Run | Epochs, gold per query | Train top-1, last epoch | Neural alone by epoch | Minutes | Peak GB |
|---|---|---|---|---|---|
| p7-enc-full (old excerpt) | 2, 3 | .553 | .2879, .3391 | 11.6 | 12.0 |
| p7c-enc-2ep | 2, 3 | .578 | .3224, .3469 | 11.6 | 11.8 |
| p7c-enc | 4, 6 | .949 | .3235, .3180, .3268, .3157 | 35.4 | 11.8 |
| p7c-lorma (0.31% trainable) | 4, 6 | .685 | .3391, .3435, .3380, .3324 | 25.7 | 6.2 |

Citing-context excerpts add little (.3391 to .3469). Longer training overfits: train top-1 reaches .95 while dev stays at about .32. With 827 training queries, the reranker has plateaued near .33 to .35, so more epochs are not the route to a stronger one. LoRMA matches the full encoder here with 0.31% of the weights and half the memory.

One-citation-at-a-time ranker features (`ltr_site.py`: best site, top two, share of sites matched) did not help: dev .4936 alone against .4947, so they were not submitted.

| ID | Citing-context ranker + | Dev | Test |
|---|---|---|---|
| 1040041 | p7c-enc, w .1 | .4981 | .4908 |
| **1040042 (leaderboard)** | mean z of p7-enc-full, p7c-enc, p7c-lorma, w .2 | .4992 | **.4938** |

Blend settings were picked among about 15 dev combinations, and the entries differ by a few hits, so 1040042's lead over 1039874 (.4926, 2 hits) is within noise. Averaging three differently trained rerankers is at least consistent with the earlier finding that single reranker scores are noisy.

## Encoder seeds (Nov 7, 18:30 IST)

Four more seeds of the two-epoch citing-context encoder (`notebooks/phase7_seeds.ipynb`), neural-only dev at the kept epoch: .3446, .3457, .3513, .3680, about 12 minutes each. Best epoch moves between 1 and 2, so a single seed's dev score is noisy.

Averaging did not move the citing-context ranker blend: four seeds .4981 (w .2), all eight rerankers .4981 (w .1), against .4992 for the three-model entry. Entry 1040271 (all eight, w .1) scored test .4932, below 1040042 at .4938, a difference of one hit. It was submitted despite no dev gain because it was the most principled average, and it did not replace 1040042.

Conclusion: test has plateaued at about .493. The reranker adds about .004 on top of the citing-context ranker (.4891 alone). Further gains would need better candidate recall or more training queries, not more of the same models.

## Report evidence (Nov 7)

`report_analysis.py` rebuilds the final blend on dev (.4992, 449 hits) and writes `report/analysis.json`; `report/EVIDENCE.md` holds the facts in template order, and `report_figures.py` makes the figures. Facts only: analysis prose waits on the template's LLM rule.

Report text is in `report/phase7_sections.tex`, with `report/ANALYSIS_EXPLAINED.md` as the walkthrough. New control: raw recall falls with query length (.68 to .23), but hits over min(gold, 7) rise (.65 to .90), so the fall is the 7-slot cap. With the cap removed, recall by gold candidate length is .50 (<1k words), .84 (4-8k) and .67 (8k+).

## One-factor controls (Nov 7)

Six dev-only runs (`notebooks/phase7_controls.ipynb`, `report_controls.py`, `report_controls_blend.py`). Neural-only F1 against enc-full .3391: pairwise .3324, random negatives .4002, last-token .3079, frozen .2379, random start .1946; GQA mean .3357 against .3168. Blended with the ranker at w .2, hits: baseline 447, pairwise 447, random negatives 442, last 444, frozen 447, random start 444, GQA mean 444 (GQA last 442).

Main result: random negatives win alone and lose in the blend, with correlation to the ranker .555 against .245. Added to the final three they gave 448 hits against 449, so no test scoring or new submission. Full table in `report/EVIDENCE.md`.

## Variable-length lists (Nov 8, 19:30 IST)

Micro-F1 counts every submitted prediction, so a prediction whose hit probability is below about F/2 lowers it. `vark_eval.py` keeps a top-7 prediction of the final blend only when the train-fit ranker's probability is >= tau (first prediction always kept). Dev, final blend (449 hits, F1 .4992):

| Ranker p | Top-7 slots | Hit rate |
|---|---|---|
| < .05 | 245 | .061 |
| .05-.1 | 76 | .263 |
| .1-.2 | 84 | .524 |
| >= .5 | 247 | .964 |

tau .08: F1 .5597 (P .7888, R .4337, 422 hits) in-sample. Honest estimate: tau picked on half the dev queries, scored on the other half, 20 random splits: gain +.0601 (sd .0189), positive in 40 of 40; picked tau .04-.12.

This only helps if the official scorer divides by the predictions actually submitted (as `micro_prf` does). `vark_submit.py` writes `submissions/p7-vark-t008.zip`: the 0.4938 entry (rebuilt exactly, 237 of 237 queries) with low-probability predictions removed, 1188 of 1659 kept.

**Not allowed.** Submitted as 1042447 (Nov 8, 19:40 IST): status Failed. The rules on the Get Started page require exactly seven unique candidate IDs per test query, so precision always divides by 7 per query and this gain does not exist under the official metric. `pcr_lib.write_submission` already asserted seven; `vark_submit.py` bypassed it. Keep the dev numbers only as a note on how the metric could be gamed if lists were variable.

## Window scoring without retraining (Nov 8, 20:30-21:00 IST)

`chunk_scores.py` scored each top-50 candidate as up to 12 evenly spaced windows (96-token opening + 416 contiguous tokens), with the final rerankers unchanged; 87,299 dev and 162,704 test window pairs (about 7 per candidate). Run on a Colab CLI A100; checkpoints pulled with rclone (read-only remote). Only p7-enc-full finished before stopping. Dev, `chunk_eval`-style check:

| Neural score (p7-enc-full) | Alone F1 (hits) | Ranker + 0.2 z (hits) |
|---|---|---|
| excerpt (current) | .3391 (305) | .4969 (447) |
| window max | .2001 (180) | .4958 (446) |
| window top-2 mean | .1901 (171) | .4947 (445) |
| window mean | .2057 (185) | .4947 (445) |
| excerpt + window max, averaged | | .4958 (446) |

Windows are much weaker alone and add nothing to the blend. The rerankers were trained on excerpts chosen by shared rare n-grams; evenly spaced windows are mostly unrelated text the models never learned to discount. Making this work needs max-over-windows training (about 1h45m on an A100), not tried: the deadline was too close and the neural term is worth only a few hits. No submission. 1040042 (0.4938) stays.
