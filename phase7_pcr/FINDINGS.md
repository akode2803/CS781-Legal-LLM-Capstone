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
| `p7-shingle4-b06-hdrdate.zip` (1038437, on leaderboard) | 0.4554 | 0.4647 | 0.4464 | 0.4603 |

## Notes

- Gold candidates are always older than the query when both have a `DATE OF JUDGMENT` header (352 of 352 pairs), so dropping later candidates is safe. The header parses for about half of the documents.
- Highest year in the text is not a usable date: 20-33% of gold candidates violate it.
- Mixing n-gram sizes by z-score did not beat the best single size.

## Citation traces (separate, disclosed experiment)

The organisers masked citations, but traces survive: header `CITATION:` and `CITATOR INFO` blocks, and reporter references in footnotes. See [citation_residue.py](citation_residue.py).

| Lexical ranking, matched candidates moved to the front | Dev F1 | Train F1 |
|---|---|---|
| No promotion | 0.4603 | 0.3760 |
| Forward matches (precision 0.91 dev, 0.71 train) | 0.4703 | 0.3938 |
| Citator matches (precision 0.98 dev, 0.91 train) | 0.4703 | 0.3974 |
| Both (precision 0.93 dev, 0.75 train) | 0.4747 | 0.4088 |

Removing the traces from the text (`scrub`, 0.5% of characters) lowers the plain lexical score slightly: dev 0.4603 to 0.4569, train 0.3760 to 0.3747. Our own match patterns find nothing afterwards; that does not prove every trace is gone.

Submission 1038437 (test 0.4554) was built on unscrubbed text, so it did not match citations explicitly but did read the traces.
