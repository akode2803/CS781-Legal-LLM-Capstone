# Phase 7 analysis, explained

This goes with `phase7_sections.tex` (trimmed for the report; some detail below is only in `phase7_sections_full.tex`, so be ready to explain it without the page). For each claim: what it means, the evidence, and what an examiner may push on. All numbers are in `EVIDENCE.md`; `report_analysis.py` regenerates the dev ones.

## How to read the numbers

- **Micro-F1@7.** Add up hits, predictions and gold across all queries, then compute F1 once. Big-gold queries therefore weigh more in recall.
- **One hit** is about .0006 test F1, so .4926 against .4938 is two hits. Treat anything that small as a tie.
- **Paired bootstrap.** Resample the 118 dev queries with replacement 5000 times and recompute the difference between two systems on the same resample. If the 95% interval excludes 0, the gain is unlikely to be luck from which queries were sampled. It does not cover the bias from choosing settings on dev.
- **AUC inside the top 50.** Pick a gold and a non-gold candidate from a query's top 50: AUC is how often the gold scores higher. .5 is chance. It uses the whole ranking, so it is less noisy than F1@7.
- **z-score per query.** Subtract the query's mean and divide by its standard deviation, so scores from different models can be added.

## The seven claims to know

**1. Exact text reuse carries most of the score.** Lexical alone gets .4554 of the final .4938. Unigram BM25 gets .10 because topic words appear in hundreds of cases; a shared 4-gram is much rarer, and judgments quote what they cite.

**2. Neural models help only as a small correction.** They are weak alone (AUC about .75 against .915), but their rank correlation with lexical order is at most .17, so they carry different information. Adding a small weight helps; replacing the lexical order hurts (.4514 or lower). *Likely question: why not make the neural model the main ranker?* It reads about 1024 tokens of a pair averaging thousands of words each, and has only 827 training queries.

**3. Adaptation methods tie once blended.** All seven are within 5 dev hits. Compare them on neural-only scores instead. The encoder over GQA full interval excludes 0, but it is one of six uncorrected comparisons, and backbone and pooling changed together, so it says nothing about pooling alone.

**4. LoRMA is the practical choice.** Same quality as full fine-tuning, 0.31% of the weights, half the memory. The data-scarcity explanation is a hypothesis, and the report says so. LoRMA multiplies the weight matrix by a low-rank update, where LoRA adds one.

**5. KD lost quality but gained speed.** Student: 8 of 16 blocks, 42% fewer parameters, 2x faster, trained on cross-entropy plus KL to the teacher's softened scores (T 2, alpha .5). Neural F1 fell by a third. Do not say "KD preserved quality": the blended score hides the loss because the neural term is small.

**6. Long queries are not the problem; the 7-slot cap is.** This is the subtlest result. Raw recall falls from .68 to .23 as queries get longer, which looks like a long-document failure. But long queries have far more gold (26 on average in the longest band), and seven slots cannot hold them. Dividing hits by min(gold, 7) reverses the trend: .65 up to .90. Caveat: many gold also makes each slot easier to fill, so this shows long queries are not harder, not that they are easier for the model. Long and very short *candidates* do show a real dip, even with the cap removed.

**7. Remaining errors are mostly citations without shared text.** Hits have overlap z 3.37, missed gold .56. The neural term could fix these in principle, but it swapped only 58 of 826 entries, and its excerpts are chosen by the same n-gram overlap. That link is a hypothesis.

**8. The controls: a better model alone can be a worse part of the system.** Read these against the seed spread: s.d. about .01 neural F1, so anything under about .02 is noise.

- *Transfer:* random start .1946, frozen .2379, full .3391, chance .1124. Frozen trains only a 1,025-parameter head and still beats a fully trained random start, so Phase 1 weights carry usable signal, and fine-tuning adds more.
- *Pooling:* mean beats last on the encoder (-.0311, interval excludes 0). This is expected: an MLM encoder has no reason to summarise a document into its final token. On GQA, mean is +.0189 but the interval includes 0, so call it unresolved, even though a causal model's last token is the usual choice.
- *Backbone:* claim 3's encoder-over-GQA gap mostly disappears once pooling matches (.3391 against .3357), so don't say the encoder is the better backbone here.
- *Loss:* pairwise against softmax, -.0067, no difference.
- *Negatives:* random negatives are the surprise. Neural F1 rises to .4002 (+.0611), but the blend drops to 442 hits from 447. Why: telling a gold case from a random case teaches broad topic match. That also lifts scores inside the top 50, but the ranker already knows it, as the correlation with the ranker shows (.555 against .245). Hard negatives share vocabulary with the query, so the model must use something else. It scores worse alone but adds more. If asked "why not use the better model?": the deployed quantity is the blend, and the blend got worse.
- *Blend resolution:* frozen ties full at 447, and the whole range is 442 to 447, or 5 hits. The blend can't rank these models, which is why claims about them use neural-only scores.

## Weak points to admit first

- Each control has one seed. Random start was tried at one learning rate (1e-4) for two epochs, so "Phase 1 helps" holds for this budget, not in general.
- Phase 1 transfer shows up in standalone scores only. In the blend, a frozen model ties full fine-tuning at 447 hits, so don't claim fine-tuning caused the blended gain.
- "Random negatives learn what the ranker already knows" is an interpretation. The evidence is one correlation (.555 against .245).
- Blend settings were picked on dev from about 15 combinations.
- The ranker and citing-context features are not Phase 1 models.
- `<CITATION>` position is used, which you approved as different from case identity. Traces stay in the text, unscrubbed, and our features read them indirectly. Expect a question on this; the answer is that deliberate matching was measured, rejected and never submitted.

## Before submitting

- Declaration Q2: tuning, analysis, core code and report writing all used an LLM, so those rows must be Yes. Q4 must be in your own words, and it asks for two failures: use neural-as-feature and four-epoch training.
- Check the LoRMA author names: the capstone document's initials differ from `phase7_refs.bib`, which I filled in from memory. The arXiv id 2506.07621 comes from the document.
- Fill in the final rank after 23:59 IST.
