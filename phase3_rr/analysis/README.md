# Phase 3 offline analysis (CPU only)

Each script takes one argument: a folder holding the Phase 3 files `CL_train.json CL_dev.json IT_train.json IT_dev.json`
(download them from the Hugging Face dataset with `common/download_data.py phase3`).

| script | what it answers |
|---|---|
| `data_analysis.py` | label share, document lengths, sentence token lengths (truncation at 63), label persistence and the copy-previous-label ceiling |
| `tfidf_baseline.py` | TF-IDF + logistic regression with neighbour and position features; then count-based Viterbi decoding (which made things worse) |
| `doc_head_cv.py` | grouped 5-fold CV of document-level BiLSTM / CRF / label-shift heads on TF-IDF-SVD features (uses `../doclib.py`) |

Reported results (Phase 3 dev, present-class macro-F1): TF-IDF sentence only 0.312; + position 0.335; + neighbours ±1 0.396;
document BiLSTM on TF-IDF vectors (CV) 0.357 vs sentence MLP 0.319.
