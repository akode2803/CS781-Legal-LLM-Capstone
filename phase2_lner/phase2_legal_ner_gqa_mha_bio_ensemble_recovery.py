# One-cell recovery for the already-open Colab runtime after the endpoint-parity error.
# It preserves the old (truncated) cache and writes fresh caches under cachefix_v2.
# Because the old cache kept only each window's first-word logits, the missing
# word logits cannot be reconstructed; this reloads both checkpoints sequentially
# and performs inference only (no training, no runtime allocation).
import importlib.util, shutil

assert 'cache_one' in globals() and 'val_windows' in globals() and 'val_rows' in globals(), \
    'Run this in the existing ensemble notebook runtime after the parity error.'
shutil.copy2(ENSEMBLE_HELPER, LOCAL / 'bio_ensemble.py')
spec = importlib.util.spec_from_file_location('bio_ensemble_recovered', LOCAL / 'bio_ensemble.py')
ensemble_fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ensemble_fixed)
# cache_one resolves these global names when called; explicitly rebind them so
# the stale `from bio_ensemble import *` definitions cannot be reused.
for name in ('cache_key', 'collect_emission_cache', 'blend_components',
             'evaluate_weight', 'viterbi_decode'):
    globals()[name] = getattr(ensemble_fixed, name)

OLD_RUN_ROOT = RUN_ROOT
RUN_ROOT = OLD_RUN_ROOT / 'cachefix_v2'
gqa_val, gqa_provenance = cache_one('gqa', 'validation', val_windows)
mha_val, mha_provenance = cache_one('mha', 'validation', val_windows)
validation_results = {}
for weight in GQA_WEIGHT_GRID:
    result = evaluate_weight(val_windows, val_rows, gqa_val, mha_val, weight,
                             LABELS, TAGS, decode_entities)
    if weight in (0.0, 1.0) and not result['endpoint_decoder_parity']:
        raise AssertionError(f'endpoint decoder parity failed after recache at GQA weight {weight}')
    validation_results[weight] = result
    print(f'GQA/MHA={weight:.2f}/{1-weight:.2f}; macro-F1={result["macro_f1"]:.12f}; '
          f'parity={result["endpoint_decoder_parity"]}; '
          f'cached CUDA-path differences={result["cached_reference_path_device_mismatches"]}')

validation_winner_weight = max(GQA_WEIGHT_GRID, key=lambda w: validation_results[w]['macro_f1'])
baseline_f1 = validation_results[1.0]['macro_f1']
winner_f1 = validation_results[validation_winner_weight]['macro_f1']
compact = {str(w): {k: v for k, v in r.items() if k != 'predictions'}
           for w, r in validation_results.items()}
report = {
    'validation_documents': 6, 'validation_gold_entities': 781,
    'document_ids': [str(r['id']) for r in val_rows],
    'checkpoint_provenance': {'gqa': gqa_provenance, 'mha': mha_provenance},
    'gqa_weights': list(GQA_WEIGHT_GRID), 'results': compact,
    'gqa_baseline_macro_f1': baseline_f1,
    'validation_winner_gqa_weight': validation_winner_weight,
    'validation_winner_macro_f1': winner_f1,
    'validation_winner_beats_gqa': winner_f1 > baseline_f1,
    'test_used_for_selection': False,
    'cache_recovery': 'schema v2; full [words,tags] emissions; old cache preserved',
}
RUN_ROOT.mkdir(parents=True, exist_ok=True)
result_path = RUN_ROOT / 'validation_results.json'
tmp_path = result_path.with_suffix('.json.tmp')
tmp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
tmp_path.replace(result_path)
print('Fresh validation comparison saved:', result_path)
