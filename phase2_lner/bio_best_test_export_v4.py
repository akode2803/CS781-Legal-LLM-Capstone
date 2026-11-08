"""Post-training BIO test export cell (pinned Phase 2 test edition).

Run only after the BIO notebook setup/model cells have defined the referenced
names.  This deliberately loads best.pt and writes a separate submission.
"""

from pathlib import Path
import json
import zipfile

import torch


TEST_DIR = COURSE_ROOT / "Phase_2_LNER/Test"
BEST_CHECKPOINT = RUN_ROOT / "best.pt"
SUBMISSION_DIR = RUN_ROOT / "submission_best"
TEST_REPOSITORY = "Exploration-Lab/CS-781-Capstone"
TEST_REVISION = "89edba6a44d6bb9593fd13b6895b5cd3b06eb82e"
TEST_REPOSITORY_PATH = "Phase_2_LNER/Test/fold3-text.json"
# Phase 1's official-test/test.json is unrelated and must never be substituted.
test_files = [COURSE_ROOT / TEST_REPOSITORY_PATH]
missing_test_files = [path for path in test_files if not path.is_file()]
if missing_test_files:
    raise FileNotFoundError(
        "Pinned Phase 2 NER test file is not readable locally. Download only "
        f"{TEST_REPOSITORY_PATH} from {TEST_REPOSITORY} at revision {TEST_REVISION} "
        f"with the approved authenticated Hugging Face session: {missing_test_files}"
    )

if not BEST_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Best BIO checkpoint not found: {BEST_CHECKPOINT}")

payload = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
if payload.get("schema_version") != 1 or payload.get("identity") != IDENTITY:
    raise ValueError("Best BIO checkpoint identity mismatch")
model.load_state_dict(payload["model"], strict=True)

test_rows = []
for path in test_files:
    test_rows.extend(load_records(path))
if not test_rows:
    raise ValueError("Unlabeled test files contained no documents; refusing empty submission")
if len({row["id"] for row in test_rows}) != len(test_rows):
    raise ValueError("Duplicate official document IDs across test files")
empty_text_ids = [row["id"] for row in test_rows if not row["text"]]

test_windows = [
    window
    for row in test_rows
    if row["text"]
    for window in make_document_windows(row, TOKENIZER, MAX_BPE, STRIDE_BPE)
]

if not torch.cuda.is_available():
    raise RuntimeError("BIO test inference requires CUDA; refusing silent CPU fallback")
device = "cuda"
model.to(device).eval()
decoded = []
with torch.inference_mode():
    for window in test_windows:
        ids = torch.tensor([window["input_ids"]], dtype=torch.long, device=device)
        last = torch.tensor(window["word_last"], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            emissions = model.emissions(ids, last)
        decoded.extend(decode_entities(window, model.crf.decode(emissions.float()), model.tags))

# export_prediction is the existing official-schema writer.  BIO/CRF decoding
# has no confidence threshold, so the unused score field is a fixed sentinel.
unique_predictions = sorted(set(decoded), key=lambda item: (item[0], item[1], item[2], item[3]))
scored_predictions = [(*item, 1.0) for item in unique_predictions]
texts = {row["id"]: row["text"] for row in test_rows}
prediction_path, zip_path = export_prediction(
    [row["id"] for row in test_rows], scored_predictions, texts, SUBMISSION_DIR
)

written = json.loads(prediction_path.read_text(encoding="utf-8"))
if list(written) != [row["id"] for row in test_rows]:
    raise ValueError("Export changed official test ID order or membership")
with zipfile.ZipFile(zip_path) as archive:
    if archive.namelist() != ["prediction.json"]:
        raise ValueError("Submission ZIP must contain only root prediction.json")

print("Loaded best checkpoint:", BEST_CHECKPOINT)
print("Best validation progress:", payload["progress"])
print("Test files:", [str(path) for path in test_files])
print("Documents:", len(test_rows), "decoded entities:", len(unique_predictions))
print("Empty official documents retained with []:", empty_text_ids)
print("Submission JSON:", prediction_path)
print("Submission ZIP:", zip_path)
