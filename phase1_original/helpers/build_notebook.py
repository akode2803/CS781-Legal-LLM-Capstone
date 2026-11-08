"""One visible run path, reusing existing readable helpers and frozen tokenizer."""
from pathlib import Path
import hashlib,json,uuid,zipfile
root=Path(__file__).parent
version=json.loads((root/'source-version.json').read_text())
version['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'legal_scaffold').glob('*.py'))}
(root/'source-version.json').write_text(json.dumps(version,indent=2)+'\n')
archive=root/'legal_course_helpers_v1.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for folder in ('legal_scaffold','references','tokenizer','tests'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(root).as_posix())
    for name in ('source-version.json','amp-overflow-compatibility.json','README.md','build_notebook.py'):z.write(root/name,name)
cells=[]
def add(kind,s):
    value={'cell_type':kind,'id':uuid.uuid4().hex[:8],'metadata':{},'source':s.splitlines(True)}
    if kind=='code':value.update(execution_count=None,outputs=[])
    cells.append(value)
add('markdown','''# Legal LLM course — one run path

**Use this notebook and the small `legal_course_helpers_v1.zip` together.** The ZIP contains ordinary readable Python, the existing frozen BPE and source notices. Its tokenizer/ directory already supplies the frozen BPE; the older sample/canonical preparation ZIP is not needed. Every stage is visible below: choose an existing architecture, prepare complete WikiText TRAIN or load a completed stage, continue on the four legal sources, train cumulative 100M, 200M and later legal-token milestones, then export each milestone prediction ZIP. Corrections for AMP overflow, the one verified WikiText TRAIN/Test duplicate, and empty legal text handling are already included. Legal preparation now tokenizes the raw files without deduplication and resumes from committed small token shards.

This reuses your existing model and training code without changing its architecture or optimization settings. Experimental choices, model quality, results analysis and conclusions remain yours. Review and declare AI assistance under the instructor permission you reported. No pretrained weights, automatic training, paid activation, external sharing or Codabench submission is performed.

Colab: upload this notebook and the helper ZIP, use a fresh runtime, and mount Drive below. Kaggle: import this notebook and attach the helper files as a private Input. If Kaggle extracts the ZIP, set HELPER_DIRECTORY to the folder containing legal_scaffold/, tokenizer/ and source-version.json; no rezipping or reupload is needed. Leave HELPER_ZIP unset when using the directory route. Required packages are `torch`, `tokenizers`, and `pyarrow` for WikiText preparation; `huggingface_hub` is needed only for optional gated downloads. Install/select packages yourself if absent. Prepare on CPU where practical; choose **one GPU** yourself before training/prediction. Re-run the setup/config cells after switching runtimes. Keep running preparation and WikiText training in their current runtime. Start legal milestones in a fresh runtime to avoid cached older helper imports. The ZIP route checks the extracted source manifest and frozen tokenizer; there is no hard-coded ZIP byte hash.
''')
add('code',rf'''from pathlib import Path
import hashlib, json, math, os, sys, tempfile, uuid, zipfile, importlib.util, shutil

PLATFORM = ("kaggle" if Path("/kaggle/working").is_dir() else
            "colab" if Path("/var/colab/hostname").is_file() else "local")
# For a custom runtime, set PLATFORM explicitly to "kaggle", "colab" or "local".

HELPER_DIRECTORY = None  # Kaggle: set the folder containing legal_scaffold/, tokenizer/, source-version.json.
HELPER_ZIP = None        # Optional ZIP route; leave None when using HELPER_DIRECTORY.
PACKAGE_ROOT = (Path('/kaggle/working') if PLATFORM == 'kaggle' else Path.cwd()) / 'legal-course-runtime'
PACKAGE_ROOT = PACKAGE_ROOT.resolve()

if HELPER_DIRECTORY:
    source = Path(HELPER_DIRECTORY)
    if not all((source / name).exists() for name in ('legal_scaffold', 'tokenizer', 'source-version.json')):
        raise ValueError('HELPER_DIRECTORY must be the extracted helper root containing legal_scaffold/, tokenizer/, source-version.json.')
    shutil.copytree(source, PACKAGE_ROOT, dirs_exist_ok=True)
else:
    if not HELPER_ZIP and PLATFORM == 'colab':
        from google.colab import files
        uploaded = files.upload()
        assert len(uploaded) == 1
        HELPER_ZIP = next(iter(uploaded))
    assert HELPER_ZIP, 'Set HELPER_DIRECTORY to the extracted Input root, or supply HELPER_ZIP.'
    with zipfile.ZipFile(HELPER_ZIP) as archive:
        assert sum(i.file_size for i in archive.infolist()) < 8_000_000
        assert all(not Path(i.filename).is_absolute() and '..' not in Path(i.filename).parts and '\\' not in i.filename and (i.external_attr >> 16) & 0o170000 != 0o120000 for i in archive.infolist())
        archive.extractall(PACKAGE_ROOT)

sys.path.insert(0, str(PACKAGE_ROOT))
assert importlib.util.find_spec('torch') and importlib.util.find_spec('tokenizers'), 'Select/install PyTorch and tokenizers yourself.'
from legal_scaffold.source_version import verify_runtime
from legal_scaffold.core import strict_json, file_digest
from legal_scaffold.tokenizer_tools import tokenizer_identity
from legal_scaffold.snapshots import verify_snapshot
print('Source:', verify_runtime(PACKAGE_ROOT))
TOKENIZER_DIRECTORY = PACKAGE_ROOT / 'tokenizer'
TOKENIZER_IDENTITY = tokenizer_identity(TOKENIZER_DIRECTORY)
assert TOKENIZER_IDENTITY['tokenizer_sha256'] == '906194848c1632431aac7d186e7c4f49da83cb6cbd480fa222c8472401cbf8ab'
print('Frozen tokenizer vocabulary:', TOKENIZER_IDENTITY['actual_vocab_size'])
''')
add('markdown','''## 1. Choose the architecture and durable paths

Select `gqa_pre`, `mla_pre`, `mha_pre` or `mha_post`. These select the existing configurations, with unchanged shared optimizer settings. Use a distinct run ID for each architecture. Keep that ID unchanged for resume. `/kaggle/working` is session-local until you save an output version and verify it; mounted Drive is the Colab destination. Preserve complete snapshot directories, not just `state.pt`.

If you already have a completed full WikiText stage, set `WIKITEXT_STAGE` and skip WikiText training. Recovery and 100-window pilot snapshots do not qualify as completed stages. Keep an active older run in its current notebook/source until completion; this notebook can consume its verified completed stage. Exact resumes below require this release's unchanged source/data/configuration/runtime. The integrated AMP compatibility guard can accept an original pre-patch recovery only when its remaining identity matches exactly; it does not migrate arbitrary versions.
''')
add('code','''import torch
from legal_scaffold.long_training import DEFAULT_MODEL, DEFAULT_RUN, train as train_wikitext
from legal_scaffold.models import ModelConfig, describe
ARCHITECTURE = "gqa_pre"
RUN_ID = "run-001"
CHOICES = {"gqa_pre": dict(attention="gqa", kv_heads=2, norm_placement="pre"),
           "mla_pre": dict(attention="mla", kv_heads=2, norm_placement="pre", mla_latent=256, mla_query_rank=256, mla_rope_dim=32),
           "mha_pre": dict(attention="mha", kv_heads=8, norm_placement="pre"),
           "mha_post": dict(attention="mha", kv_heads=8, norm_placement="post")}
MODEL_CONFIG = dict(DEFAULT_MODEL, **CHOICES[ARCHITECTURE])
print(describe(ModelConfig(**MODEL_CONFIG)))
if PLATFORM == "kaggle":
    DURABLE_ROOT = Path("/kaggle/working/legal-llm-course")
elif PLATFORM == "colab":
    from google.colab import drive
    drive.mount("/content/drive")
    DURABLE_ROOT = Path("/content/drive/MyDrive/legal-llm-course")
else:
    DURABLE_ROOT = Path.cwd() / "local-legal-llm-course"
DURABLE_ROOT.mkdir(parents=True, exist_ok=True)
FULL_CACHE = DURABLE_ROOT / "full-wikitext/full-cache"  # Set an observed existing cache here to reuse it.
FULL_WINDOWS = FULL_CACHE / "frozen-train-windows"
WIKITEXT_ROOT = DURABLE_ROOT / "runs" / ARCHITECTURE / RUN_ID / "wikitext-course-v1"
LEGAL_ROOT = DURABLE_ROOT / "runs" / ARCHITECTURE / RUN_ID / "legal-token-stages-v1"
WIKITEXT_STAGE = None  # Set a completed snapshot directory, or train in section 3.
LEGAL_STAGE = None  # Set a verified legal milestone or completed stage for prediction.
WIKITEXT_RESUME = None
LEGAL_RESUME = None  # This architecture's previous legal milestone/recovery; keep fixed run ID.
LEGAL_TOKEN_TARGET = 100_000_000  # Cumulative legal tokens: then 200_000_000, 300_000_000.
''')
add('markdown','''## 2. Complete WikiText cache and frozen token windows

Set the preparation flag only if you need the full cache. It downloads about 315 MB of public pinned Parquet sources in your runtime, verifies size/hash, retains official validation/test, and excludes only the verified TRAIN duplicate of *The Hustler*. Other overlaps stop preparation. BPE is never refitted. Interrupted downloads and committed whole-document encoding can resume. Reuse a completed cache unchanged, especially for an existing checkpoint. Raw source and original converted files are preserved.

Allow at least 8 GB free for this cache and transient checkpoints. Legal data later requires substantially more space. Nothing has been downloaded or trained on full data by this release's local checks.
''')
add('code','''PREPARE_WIKITEXT = False
if PREPARE_WIKITEXT:
    assert importlib.util.find_spec("pyarrow"), "Select/install PyArrow yourself."
    from legal_scaffold.full_wikitext import prepare_full, encode_full
    prepared = prepare_full(FULL_CACHE)
    manifest = encode_full(prepared, TOKENIZER_DIRECTORY, FULL_WINDOWS)
    print(json.dumps(manifest, indent=2))
''')
add('markdown','''## 3. Optional GPU probe and one full WikiText pass

If loading a completed stage, leave training disabled. Otherwise select one GPU and explicitly enable training. An optional synthetic dispatch probe reports actual operators; it is not a speed or full-model guarantee. MLA uses its existing MLA-specific probe. Native BF16 is used only when reported without emulation; other CUDA hardware uses FP16 with GradScaler. Overflow skips do not advance the optimizer step, scheduler, token count or cursor; repeated instability stops the run.

The existing runner uses one complete pass, with successful-step/token caps calculated from the full cache. Recovery is every 1,000 steps or 15 minutes, retaining the newest two verified recoveries. The six-hour wall guard saves recovery; it is not a completion-time promise. On resume set the observed recovery directory from `latest-wikitext_only.json`, keep all identities fixed, and use the same GPU/runtime. Completed stage directories stay immutable. Save and independently verify platform output before releasing the GPU.
''')
add('code','''VERIFY_GPU_DISPATCH = False
if VERIFY_GPU_DISPATCH:
    if MODEL_CONFIG["attention"] == "mla":
        from legal_scaffold.mla_checks import tiny_checks, gpu_dispatch
        print(json.dumps(tiny_checks(), indent=2))
        print(json.dumps(gpu_dispatch(), indent=2))
    else:
        from legal_scaffold.kernel_dispatch import run_dispatch
        print(json.dumps(run_dispatch(), indent=2))
''')
add('code','''RUN_WIKITEXT = False
if RUN_WIKITEXT:
    assert torch.cuda.is_available(), "Select one GPU yourself."
    from legal_scaffold.window_file import WindowFile
    from legal_scaffold.wikitext import source_manifest
    from legal_scaffold.tokenizer_tools import select_corpus
    manifest = strict_json((FULL_WINDOWS / "manifest.json").read_text())
    prepared_manifest = strict_json((FULL_CACHE / "prepared-full/manifest.json").read_text())
    pinned = source_manifest()
    assert prepared_manifest["provenance"] == pinned
    assert all(prepared_manifest["splits"][s]["rows"] == pinned["expected_rows"][s] for s in ("train", "validation", "test"))
    for split in ("train", "validation", "test"):
        select_corpus(FULL_CACHE / "prepared-full", "wikitext", split)
    _, source = select_corpus(FULL_CACHE / "prepared-full", "wikitext", "train")
    assert manifest["complete_train"] is True and manifest["split"] == "train"
    assert manifest["source_corpus"] == source and manifest["tokenizer"] == TOKENIZER_IDENTITY
    path = FULL_WINDOWS / "windows.jsonl"
    assert file_digest(path) == manifest["outputs"]["windows.jsonl"]["sha256"]
    windows = WindowFile(path, Path(tempfile.gettempdir()) / (uuid.uuid4().hex + ".sqlite"), vocab_size=TOKENIZER_IDENTITY["actual_vocab_size"], context_length=MODEL_CONFIG["context_length"])
    assert len(windows) == manifest["counts"]["windows"] and windows.supervised_tokens == manifest["counts"]["supervised_tokens"]
    settings = dict(DEFAULT_RUN, max_steps=math.ceil(len(windows) / DEFAULT_RUN["batch_size"]), max_tokens=windows.supervised_tokens, max_epochs=1, max_seconds=6*3600, recovery_every_steps=1000, recovery_every_seconds=900, recovery_keep=2)
    print("Window/token budgets:", len(windows), windows.supervised_tokens)
    result = train_wikitext(windows, model_config=MODEL_CONFIG, settings=settings, checkpoint_root=WIKITEXT_ROOT, name=RUN_ID, stage="wikitext_only", resume=WIKITEXT_RESUME, tokenizer_identity=TOKENIZER_IDENTITY, tokenizer_directory=TOKENIZER_DIRECTORY, data_manifest=manifest)
    print(json.dumps(result, indent=2))
    if result["progress"]["epoch"] >= 1:
        WIKITEXT_STAGE = result["snapshot"]
    else:
        print("Stopped at recovery; resume this exact run before legal continuation:", result["snapshot"])
''')
add('markdown',"## 4. Shared mixed source order and cumulative legal milestones\n\nReuse the four downloaded legal JSON files and the existing `legal-tokenize-only-v1` cache. Keep the running Colab preparation in its current notebook/runtime. Nothing here changes or retokenizes that cache. Its frozen BPE, deterministic 90/10 document split and exact-empty handling remain unchanged.\n\nAfter all four sources finish, fill `SOURCE_END_GROUPS` from the existing source-end output lines: `source_index` 1, 2, 3 or 4, `next_record` 0, and `committed_groups` N. N is the exclusive ending group for that source. For the currently observed cache, Acts ends at 12. Other boundaries are being captured; do not guess. All architectures use the same four values and shared `MIXED_ORDER` directory.\n\nThe small order index chooses the next window from the source with the fewest supervised tokens consumed, with ties in Acts, High Courts, Supreme Court, Tribunals order. Each source contributes about 25%, differing by at most one window while all have data. Windows are never repeated. The order stops when any source exhausts, rather than duplicating a smaller source. Building this index reads existing token shards once for lengths; it does not encode raw text. Four bounded shard caches prevent rereading a shard for every alternating source. Prepare the shared index on CPU before choosing a training GPU.\n\nFor each architecture, set its own completed `WIKITEXT_STAGE` and distinct `RUN_ID`. First legal training keeps the existing policy: WikiText weights, fresh legal AdamW/scheduler/scaler with unchanged optimization settings. Set `LEGAL_TOKEN_TARGET = 100_000_000`. Then set `LEGAL_RESUME` to that model's resulting checkpoint and target 200M, 300M and so on. The target is outside the fixed resume identity; model, optimizer settings, data order, code and GPU/runtime remain fixed. Later legal stages restore optimizer/scaler/scheduler/RNG and the next window; they do not reload WikiText weights or reset the data cursor.\n\nTargets count actual supervised tokens in successful optimizer updates. The batch crossing a target finishes, so at batch size 2 and 1024 supervised tokens/window the overshoot is at most 2,047 tokens. Receipts/logs record actual total, overshoot, and cumulative per-source exposure. Failed AMP steps do not count. Milestones are immutable and are never recovery-pruned. A time guard before the target saves a recovery, which must be resumed before exporting. Reaching a token target does not mark an epoch complete. The one-pass flag refers only to exhaustion of this fixed balanced order, not every raw source document.\n")
add('code','''from legal_scaffold.legal_data import FILES, download_sources
from legal_scaffold.legal_tokenize_only import prepare
from legal_scaffold.legal_mix import MixedTokenShards
LEGAL_PATHS = {name: None for name in FILES}
# acts: Phase_1_Pretraining/Train/train_acts.json
# high_courts: Phase_1_Pretraining/Train/train_high_courts.json
# supreme_court: Phase_1_Pretraining/Train/train_supreme_court.json
# tribunals: Phase_1_Pretraining/Train/train_tribunals.json
DOWNLOAD_LEGAL = False
if DOWNLOAD_LEGAL:
    assert importlib.util.find_spec("huggingface_hub"), "Select/install huggingface_hub yourself."
    if PLATFORM == "kaggle":
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    elif PLATFORM == "colab":
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
    else:
        raise ValueError("Supply local files outside Colab/Kaggle.")
    try:
        LEGAL_PATHS = download_sources(DURABLE_ROOT / "hf-cache", hf_token)
    finally:
        del hf_token
LEGAL_WINDOWS = DURABLE_ROOT / "legal-tokenize-only-v1"
PREPARE_LEGAL = False
if PREPARE_LEGAL:
    assert all(LEGAL_PATHS.values()), "Supply the four paths or enable gated download."
    local_work = Path("/kaggle/working") if PLATFORM == "kaggle" else Path("/content") if PLATFORM == "colab" else None
    manifest = prepare(LEGAL_PATHS, TOKENIZER_DIRECTORY, LEGAL_WINDOWS, local_work=local_work)
    print("Completed token preparation:", manifest["counts"])

# Source-end committed_groups values; the current observed Acts cache ends at 12.
SOURCE_END_GROUPS = {"acts": 12, "high_courts": None, "supreme_court": None, "tribunals": None}
MIXED_ORDER = DURABLE_ROOT / "legal-equal-token-order-v1"
PREPARE_MIXED_ORDER = False
if PREPARE_MIXED_ORDER:
    windows = MixedTokenShards(LEGAL_WINDOWS, SOURCE_END_GROUPS, MIXED_ORDER)
    print("Balanced available supervised tokens:", windows.supervised_tokens)
    print("Available/selected tokens by source:", windows.manifest["source_available_tokens"], windows.manifest["source_consumed_tokens"])
    windows.close()
''')
add('code','''RUN_LEGAL = False
if RUN_LEGAL:
    assert torch.cuda.is_available() and WIKITEXT_STAGE, "Select one GPU and supply a completed full WikiText stage."
    from legal_scaffold.legal_training import train as train_legal
    parent = verify_snapshot(WIKITEXT_STAGE)
    assert parent["stage"] == "wikitext_only" and parent["role"] == "stage" and parent["progress"]["epoch"] >= 1
    assert parent["identity"]["data_manifest"]["complete_train"] is True
    assert parent["identity"]["model"] == MODEL_CONFIG and parent["identity"]["tokenizer_identity"] == TOKENIZER_IDENTITY
    manifest = strict_json((LEGAL_WINDOWS / "manifest.json").read_text())
    assert manifest["identity"]["tokenizer"] == TOKENIZER_IDENTITY
    assert (MIXED_ORDER / "manifest.json").exists(), "Prepare the shared mixed order on CPU first."
    windows = MixedTokenShards(LEGAL_WINDOWS, SOURCE_END_GROUPS, MIXED_ORDER)
    training_manifest = {"tokenization": manifest, "training_order": windows.manifest}
    settings = dict(parent["identity"]["settings"], max_steps=math.ceil(len(windows) / parent["identity"]["settings"]["batch_size"]), max_tokens=windows.supervised_tokens, max_epochs=1, max_seconds=6*3600, recovery_every_steps=1000, recovery_every_seconds=900, recovery_keep=2)
    result = train_legal(windows, model_config=MODEL_CONFIG, settings=settings, checkpoint_root=LEGAL_ROOT, name=RUN_ID, stage="legal_continued", initial_weights=None if LEGAL_RESUME else WIKITEXT_STAGE, resume=LEGAL_RESUME, tokenizer_identity=TOKENIZER_IDENTITY, tokenizer_directory=TOKENIZER_DIRECTORY, data_manifest=training_manifest, target_tokens=LEGAL_TOKEN_TARGET)
    windows.close()
    print(json.dumps(result, indent=2))
    if result["target_reached"] or result["one_pass_complete"]:
        LEGAL_STAGE = result["snapshot"]
    else:
        print("Stopped before the cumulative target; set LEGAL_RESUME and keep LEGAL_TOKEN_TARGET unchanged:", result["snapshot"])
''')
add('markdown','''## 5. Official test → continuation-only prediction ZIP

Set the observed official `Phase_1_Pretraining/Test/test.json` path and this architecture's verified legal milestone or completed stage. Milestone exports keep the actual cumulative supervised-token count, target, overshoot, source exposure and epoch status in their receipt. Each milestone uses a separate output directory; prior predictions remain preserved. Download the test file yourself using the same gated access, or enable the small explicit download below. It is used only for inference. The exporter requires 1,000 exact IDs and 512 whitespace-separated prompt words per record. Greedy generation uses the original model cache and generates continuation tokens only; it outputs the first 128 complete continuation words. It fails if the model context/generation cap cannot supply them, without padding or prompt truncation. This format check cannot promise model quality.

The ZIP contains only root `prediction.json`: every exact ID maps to one `text` field, with nonempty 128-word continuation. Duplicate JSON keys, missing/extra IDs and other fields are rejected. Receipts and ID lists stay outside the ZIP. Download/verify the final ZIP yourself before any manual Codabench submission; submission limits remain your responsibility.
''')
add('code','''OFFICIAL_TEST = None  # Set untouched local test.json.
DOWNLOAD_TEST = False
if DOWNLOAD_TEST:
    from legal_scaffold.legal_data import REPO, REVISION, TEST
    from huggingface_hub import hf_hub_download
    if PLATFORM == "kaggle":
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    elif PLATFORM == "colab":
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
    else:
        raise ValueError("Supply local test.json outside Colab/Kaggle.")
    try:
        OFFICIAL_TEST = hf_hub_download(REPO, TEST, repo_type="dataset", revision=REVISION, cache_dir=str(DURABLE_ROOT / "hf-cache"), token=hf_token)
    finally:
        del hf_token
RUN_PREDICTION = False
EXPORT_ID = "export-001"  # Change only to make a separate repeat export; old outputs stay preserved.
if RUN_PREDICTION:
    assert torch.cuda.is_available() and LEGAL_STAGE and OFFICIAL_TEST
    from legal_scaffold.legal_prediction import export
    selected = verify_snapshot(LEGAL_STAGE)
    PREDICTION_ROOT = DURABLE_ROOT / "predictions" / ARCHITECTURE / RUN_ID / ("tokens-" + str(selected["progress"]["tokens_seen"])) / EXPORT_ID
    result = export(LEGAL_STAGE, OFFICIAL_TEST, TOKENIZER_DIRECTORY, PREDICTION_ROOT)
    print(json.dumps(result, indent=2))
    print("Only this ZIP is the submission artifact:", PREDICTION_ROOT / "submission.zip")
''')
add('markdown','''For each remaining architecture, change `ARCHITECTURE`, use its own `RUN_ID`, and follow the same ordered stages. Reuse the verified data caches and frozen BPE across architectures. Keep each model's original WikiText stage and every legal token milestone, source helpers, metadata and metric logs in Drive or verified saved Kaggle outputs. The notebook provides no BERTScore/ROUGE/BLEU analysis or experimental conclusions. CPU smoke checks used synthetic data only; actual legal schema, full GPU memory/runtime and official submission acceptance remain to be checked in your runtime.
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
(root/'legal_course_end_to_end_v1.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
