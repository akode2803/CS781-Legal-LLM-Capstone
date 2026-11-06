# Phase 6 (abstractive summarization): findings

Final leaderboard entry: hybrid, 1536 document tokens, 4 beams, no-repeat 3, length ratio 0.18. It scored **0.8450** BERTScore on test (Codabench 1030690), rank 34 of 50 on Nov 3.

## Test scores (Codabench, BERTScore F1)

| Submission | Test | Val (100 docs) |
|---|---|---|
| lead baseline | 0.8256 | 0.8337 |
| hybrid, nr6, ratio 0.22 | 0.8329 | 0.8378 |
| hybrid, nr3, ratio 0.22, greedy | 0.8356 | 0.8431 |
| hybrid, nr3, ratio 0.18, 4 beams | 0.8409 | 0.8469 |
| hybrid, nr3, ratio 0.18, 6 beams | 0.8409 | 0.8478 |
| **hybrid, DOC_BUDGET 1536, 4 beams** | **0.8450** | **0.8492** |

Test runs 0.005–0.008 below validation, probably because test references are longer (932 vs 751 words on average). Gains on validation carried over to test, apart from 6 vs 4 beams.

## What we learned

1. **Know the metric.** The scorer uses BERTScore with roberta-large, layer 17, no idf and no rescaling. It truncates inputs to 512 roberta tokens, so only the first ~400 words of a prediction and of a reference count. Text after that only affects ROUGE.
2. **No-repeat n-gram size.** No-repeat 3 beat 4, 5 and 2 on validation. With 6, the model looped by paraphrase (13% repeated word 4-grams). With 2, common pairs like "of the" are banned, and ROUGE-L fell to 0.205.
3. **Length.** Forcing longer summaries (lo 300–500, hi 900) hurt both ROUGE and BERTScore. A slightly shorter target (ratio 0.18) helped.
4. **Beam search.** 4 beams gave +0.005 on test over greedy decoding. 6 beams added nothing.
5. **Context split.** The context holds 2048 tokens in total, shared between document and summary. Giving the document 1536 tokens (and the summary ~510, about 390 words, BERTScore's window) beat 1024 / 1024, even though ROUGE fell. More source text helps the part of the summary the metric reads.
6. **Fine-tuning curves.** Validation loss was flat after epoch 1 (1.7488 → 1.7478) while training loss kept falling, so the model was starting to overfit. More epochs would not have helped.
7. **Position-weighted loss** (`POS_WEIGHT`): this down-weights summary tokens after the first ~450 words. It trained, but there is no score: compute ran out during validation.
8. **Engineering.**
   - Hybrid beam search gets slower per document with bigger batches; use about 2–3 documents × beams per batch.
   - Under compute limits, decode the test set in saved chunks ([test_shard.py](test_shard.py)) and merge them ([merge_shards.py](merge_shards.py)).
9. **Evaluation hygiene.**
   - Keep a held-out validation split scored with the official evaluator settings.
   - Compare runs per document (mean difference ± standard error) rather than by the mean alone.
   - Expect a validation → test offset.

## Concepts to review (for the quiz)

- **Decoder-only LMs used as summarizers:** a prefix (the document), a separator, then the summary. The loss is on summary tokens only.
- **The three architectures:**
  - GQA (grouped-query attention) with a KV cache.
  - Hymba-style hybrid: attention plus a state-space model, with no cache, so generation is slow.
  - MLM encoder used as a prefix LM: bidirectional attention over the prefix, causal over the summary.
- **Fine-tuning methods:**
  - Full fine-tuning.
  - LoRA and LoRMA adapters (rank, alpha, target modules).
  - Knowledge distillation (temperature, alpha, the KL term).
- **Training schedule:** cosine learning rate with warmup, AdamW, gradient accumulation, bf16 autocast and gradient clipping.
- **Decoding:**
  - Greedy decoding.
  - Beam search with a length penalty.
  - No-repeat n-gram blocking.
  - Min/max length control by blocking EOS.
- **Metrics:**
  - ROUGE-1 and ROUGE-L (LCS, stemming).
  - BERTScore: token embeddings, greedy cosine matching, precision/recall/F1, truncation.
  - The lead baseline.
- **Long documents:**
  - Truncation strategies: head, head+tail, TF-IDF sentence selection.
  - Context budget trade-offs.
- **Experiment design:**
  - Validation vs test splits.
  - Overfitting signals.
  - Per-example paired comparison and standard error.
  - Not tuning on test labels.
