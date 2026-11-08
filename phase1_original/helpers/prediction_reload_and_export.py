# Reload the export-only fix and export the already saved milestone. No training.
import importlib.util, zipfile
with zipfile.ZipFile(HELPER_ZIP) as archive:
    archive.extract('prediction_export_fix.py', PACKAGE_ROOT)
spec = importlib.util.spec_from_file_location('prediction_export_fix', PACKAGE_ROOT / 'prediction_export_fix.py')
prediction_fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prediction_fix)
export = prediction_fix.export
print('Prediction utility loaded:', prediction_fix.VERSION)
if not REACHED:
    raise RuntimeError('A reached legal milestone is required before prediction.')
saved = strict_json((LEGAL_CHECKPOINT / 'snapshot.json').read_text())
export_root = COURSE_ROOT / 'predictions' / MODEL / RUN_ID / ('tokens-' + str(saved['progress']['tokens_seen']))
snapshot_hash = file_digest(LEGAL_CHECKPOINT / 'snapshot.json')
test_hash = file_digest(OFFICIAL_TEST)
completed = None
pending = None
for path in sorted(export_root.glob('export-*')):
    receipt_path = path / 'export-receipt.json'
    if receipt_path.is_file() and (path / 'submission.zip').is_file():
        receipt = strict_json(receipt_path.read_text())
        if receipt['snapshot_manifest_sha256'] == snapshot_hash and receipt['test_sha256'] == test_hash:
            validate_prediction(path / 'submission.zip', path / 'official-ids.json', field='text')
            completed = path
            break
    progress_path = path / 'export-progress.json'
    if progress_path.is_file():
        progress = strict_json(progress_path.read_text())
        compatible = dict(progress, policy=prediction_fix.VERSION)
        if progress.get('policy') in prediction_fix.COMPATIBLE_PART_POLICIES and compatible == {'snapshot_manifest_sha256':snapshot_hash, 'test_sha256':test_hash,
                        'tokenizer':TOKENIZER_IDENTITY, 'policy':prediction_fix.VERSION,
                        'max_new_tokens':2 * saved['identity']['model']['context_length']}:
            pending = path
if completed is None:
    if pending is not None:
        completed = pending
        print('Resuming saved continuation parts:', completed)
        export(LEGAL_CHECKPOINT, OFFICIAL_TEST, TOKENIZER_DIRECTORY, completed, resume=True)
    else:
        number = 1
        while (export_root / ('export-' + str(number).zfill(3))).exists(): number += 1
        completed = export_root / ('export-' + str(number).zfill(3))
        export(LEGAL_CHECKPOINT, OFFICIAL_TEST, TOKENIZER_DIRECTORY, completed)
SUBMISSION_ZIP = completed / 'submission.zip'
print('Prediction ZIP:', SUBMISSION_ZIP)
from google.colab import files
files.download(str(SUBMISSION_ZIP))
