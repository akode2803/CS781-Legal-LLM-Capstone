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

## Notes

- Gold candidates are always older than the query when both have a `DATE OF JUDGMENT` header (352 of 352 pairs), so dropping later candidates is safe. The header parses for about half of the documents.
- Highest year in the text is not a usable date: 20-33% of gold candidates violate it.
- Mixing n-gram sizes by z-score did not beat the best single size.
- Citation residue is not used so far. Candidate header citations found in query text give tp 118 / fp 15 on dev, and candidate CITATOR INFO naming the query gives tp 105 / fp 2. The organisers masked citations on purpose.
