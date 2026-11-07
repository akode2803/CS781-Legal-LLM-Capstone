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
