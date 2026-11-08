# User-run cell: reload bounded recovery, select saved state, resume the same run.
import importlib.util, zipfile
with zipfile.ZipFile(HELPER_ZIP) as archive:
    archive.extract('amp_recovery_fix.py', PACKAGE_ROOT)
spec = importlib.util.spec_from_file_location('amp_recovery_fix', PACKAGE_ROOT / 'amp_recovery_fix.py')
recovery_fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery_fix)
train = recovery_fix.build_train(PACKAGE_ROOT)
if MODEL == 'gqa_pre' and 'choose_gqa_checkpoint' in globals():
    RESUME, REACHED = choose_gqa_checkpoint(LEGAL_ROOT, TARGET_TOKENS)
elif 'choose_checkpoint' in globals():
    RESUME, REACHED = choose_checkpoint(LEGAL_ROOT, TARGET_TOKENS, MODEL_CONFIG, SETTINGS, TOKENIZER_IDENTITY, TRAINING_MANIFEST, WIKITEXT_STAGE, SOURCE_VERSION)
else:
    raise RuntimeError('Run the existing setup/checkpoint-selection cells first; no training has started.')
if RESUME is None:
    raise RuntimeError('No saved legal checkpoint was selected; this recovery cell will not restart from scratch.')
print('Selected preserved checkpoint:', RESUME)
if REACHED:
    LEGAL_CHECKPOINT = Path(RESUME)
    print('Target already reached. Run the common export reload cell.')
else:
    windows = MixedTokenShards(TOKEN_ROOT, SOURCE_END_GROUPS, ORDER_ROOT)
    try:
        result = train(windows, model_config=MODEL_CONFIG, settings=SETTINGS, checkpoint_root=LEGAL_ROOT,
                       name=RUN_ID, stage='legal_continued', resume=RESUME, initial_weights=None,
                       tokenizer_identity=TOKENIZER_IDENTITY, tokenizer_directory=TOKENIZER_DIRECTORY,
                       data_manifest=TRAINING_MANIFEST, target_tokens=TARGET_TOKENS)
    finally:
        windows.close()
    LEGAL_CHECKPOINT = Path(result['snapshot'])
    REACHED = result['target_reached']
    if REACHED and 'promote_reached_recovery' in globals():
        LEGAL_CHECKPOINT = promote_reached_recovery(LEGAL_CHECKPOINT, LEGAL_ROOT)
    print('Saved:', LEGAL_CHECKPOINT, '— actual legal tokens:', result['tokens_seen'])
    if not REACHED: print('Recovery saved before target. Rerun this recovery cell to continue.')
