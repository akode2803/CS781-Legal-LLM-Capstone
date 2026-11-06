"""Download a phase's files from the gated Hugging Face dataset Exploration-Lab/CS-781-Capstone.

    set -a; . /path/to/HF_TOKEN.env; set +a          # the file holds HF_TOKEN=...  (never commit it, never print it)
    python common/download_data.py <out_dir> phase1|phase3|phase4|phase5|phase6|phase7

The token is read from the environment only and is masked in any error text."""
import os, sys
from huggingface_hub import hf_hub_download

FILES = {
    'phase3': [f'Phase_3_RR/{p}' for p in ('Train/CL_train.json', 'Train/CL_dev.json', 'Train/IT_train.json', 'Train/IT_dev.json',
                                           'Train/label_vocab.json', 'Test/CL_test.json', 'Test/IT_test.json')],
    'phase4': ['Phase_4_CJPE/Train/train.json', 'Phase_4_CJPE/Train/dev.json', 'Phase_4_CJPE/Test/test.json'],
    'phase5': ['Phase_5_LSI/Train/train.json', 'Phase_5_LSI/Train/dev.json', 'Phase_5_LSI/Test/test.json'],
    'phase6': ['Phase_6_Summ/Train/train.json', 'Phase_6_Summ/Test/test.json'],
    'phase7': [f'Phase_7_PCR/{p}' for p in ('Train/train.json', 'Train/train_candidates.json', 'Dev/dev.json', 'Dev/dev_candidates.json', 'Test/test.json', 'Test/test_candidates.json')],
    'phase1': ['Phase_1_Pretraining/Train/train_acts.json', 'Phase_1_Pretraining/Train/train_supreme_court.json',
               'Phase_1_Pretraining/Train/train_tribunals.json', 'Phase_1_Pretraining/Train/train_high_courts.json', 'Phase_1_Pretraining/Test/test.json'],   # 19.2 GB
}
tok = next((os.environ[k] for k in ('HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN', 'HUGGING_FACE_HUB_TOKEN') if os.environ.get(k)), None)
if not tok or len(sys.argv) != 3 or sys.argv[2] not in FILES:
    sys.exit(__doc__)
for f in FILES[sys.argv[2]]:
    try:
        p = hf_hub_download(repo_id='Exploration-Lab/CS-781-Capstone', repo_type='dataset', filename=f, local_dir=sys.argv[1], token=tok)
        print('ok', f, os.path.getsize(p), 'bytes')
    except Exception as e:
        print('FAILED', f, type(e).__name__, str(e).replace(tok, '***')[:200])
