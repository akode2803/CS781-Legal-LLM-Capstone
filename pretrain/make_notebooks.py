"""Regenerates pretrain/notebooks/*.ipynb (edit defaults here, not in the notebooks). The notebooks embed gpu_benchmark.py / train.py, so nothing else has to be copied to Drive."""
import nbformat as nbf
from pathlib import Path

here = Path(__file__).parent
code, md = nbf.v4.new_code_cell, nbf.v4.new_markdown_cell


def benchmark():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# GPU benchmark (pretraining speed)\nRun on each GPU you want to compare. Runtime > Change runtime type, then Run all (5 to 10 minutes). Send back the printed table."),
        code("from google.colab import drive\ndrive.mount('/content/drive')\n!unzip -q -o /content/drive/MyDrive/legal-llm-course/phase2_lner_multimethod_bundle.zip -d /content/scaffold\n!ls /content/scaffold"),
        code("%%writefile /content/hybrid.py\n" + (here / 'hybrid.py').read_text()),
        code("%%writefile /content/gpu_benchmark.py\n" + (here / 'gpu_benchmark.py').read_text()),
        code("import torch, subprocess\nname=torch.cuda.get_device_name(0); gb=torch.cuda.get_device_properties(0).total_memory/2**30\n"
             "# compute units per hour, as listed in Colab on 2026-10-27\nif 'T4' in name: cu=1.07\nelif 'L4' in name: cu=1.54\n"
             "elif 'A100' in name: cu=6.77 if gb>60 else 5.3\nelse: cu=8.9   # G4 (96 GB)\nprint(name, f'{gb:.0f} GiB', 'CU/h', cu)\n"
             "subprocess.run(f'python /content/gpu_benchmark.py --scaffold /content/scaffold --cu-per-hour {cu} --micro-batches 4,8,16,32,64', shell=True)"),
        code("!python /content/gpu_benchmark.py --scaffold /content/scaffold --cu-per-hour {cu} --micro-batches 8,16,32 --compile"),
        code("# MLA decoder (materialised mode, explicit mask: expect it to be slower than GQA)\n!python /content/gpu_benchmark.py --scaffold /content/scaffold --cu-per-hour {cu} --attention mla --micro-batches 8,16,32 --compile"),
        code("# Hymba-style hybrid (attention + SSM heads in parallel), about 214M parameters\n!python /content/gpu_benchmark.py --scaffold /content/scaffold --cu-per-hour {cu} --attention hybrid --ffn 2688 --micro-batches 8,16,32 --compile"),
    ]
    nbf.write(nb, here / 'notebooks/gpu_benchmark.ipynb')


def pretrain(name, title, *, note, extra_args='', micro=32):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"# {title}\n{note}\n\nRun on a **G4** runtime (RTX PRO 6000, 96 GB). The run is resumable: if the session ends, open the notebook on a G4 again and Run all; it continues from the last checkpoint on Drive (saved every 1000 steps, about every 70 minutes).\n"
           "Needs on Drive: `legal-llm-course/phase2_lner_multimethod_bundle.zip` (scaffold) and the tokenised corpus folder `legal-llm-course/legal_tokens/` (58 shard files, manifest.json and drop_docs.json)."),
        code("RUN_NAME = '" + name + "'\nMAX_MINUTES = 0          # 0 = no limit; e.g. 600 stops cleanly after 10 hours and saves\nTOKENS_BUDGET = 4e9\nMICRO_BATCH = " + str(micro) + "         # decoder: 32 fits the 96 GB G4 (peak about 47 GB), use 16 on a 40 GB A100; encoder: 128 (same tokens per micro-batch)\n"
             "DRIVE = '/content/drive/MyDrive/legal-llm-course'\nDISCONNECT_WHEN_DONE = True   # the last cell releases the runtime when training has ended (finished, stopped or crashed) so no compute units are wasted"),
        code("from google.colab import drive\ndrive.mount('/content/drive')\nimport torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/2**30, 'GiB')"),
        code("# scaffold + tokens onto the local disk (reading shards straight from Drive during training would be far too slow)\n!unzip -q -o {DRIVE}/phase2_lner_multimethod_bundle.zip -d /content/scaffold\n"
             "!mkdir -p /content/legal_tokens && rsync -a --info=progress2 {DRIVE}/legal_tokens/ /content/legal_tokens/\n!ls /content/legal_tokens | wc -l; du -sh /content/legal_tokens"),
        code("%%writefile /content/encoder.py\n" + (here / 'encoder.py').read_text()),
        code("%%writefile /content/hybrid.py\n" + (here / 'hybrid.py').read_text()),
        code("%%writefile /content/train.py\n" + (here / 'train.py').read_text()),
        code("!python /content/train.py --tokens /content/legal_tokens --scaffold /content/scaffold --out {DRIVE}/pretrain/{RUN_NAME} --tokens-budget {TOKENS_BUDGET} "
             "--micro-batch {MICRO_BATCH} --max-minutes {MAX_MINUTES} --compile --drop-file /content/legal_tokens/drop_docs.json --drop-min 20 " + extra_args),
        code("# training / validation curves (train.py refreshes curves.png at every evaluation; run this cell from a second session to watch a live run)\nfrom IPython.display import Image, display\ndisplay(Image(f'{DRIVE}/pretrain/{RUN_NAME}/curves.png'))"),
        code("# To stop a run cleanly from ANOTHER session or the Drive web page: create an empty file named STOP in the run folder; the run saves and ends within one step.\n# (Or press the stop button on the training cell: it also saves.)\n"
             "if DISCONNECT_WHEN_DONE:\n    import time\n    from google.colab import drive, runtime\n    print('disconnecting the runtime in 60 s (interrupt this cell to keep it)'); time.sleep(60)\n    drive.flush_and_unmount(); runtime.unassign()"),
    ]
    nbf.write(nb, here / f'notebooks/{name}.ipynb')


benchmark()
pretrain('pretrain_dec_gqa', 'Pretrain 1: GQA causal decoder, 214M parameters, 4B tokens',
         note='Clean-slate (random init) next-token pretraining on the Phase 1 legal corpus, one pass, GQA (16 heads, 4 KV heads), d_model 1024, 16 layers, 2048 context.')
pretrain('pretrain_dec_mla', 'Pretrain 2: MLA causal decoder, same size and data as Pretrain 1',
         note='Identical to Pretrain 1 (same data order, seed, schedule) except attention is multi-head latent attention, so GQA vs MLA is a clean comparison at pretraining scale.', extra_args='--attention mla')
pretrain('pretrain_enc_mlm', 'Pretrain 3: bidirectional encoder (masked language modelling), 214M parameters, 4B tokens',
         note='Same architecture family as the decoder (GQA, d_model 1024, 16 layers) but attention sees the whole 2048-token window, trained to fill in 30% masked tokens (80% mask id, 10% random, 10% kept). '
         'Loading it downstream: `restore_backbone(...)` then `to_encoder(model, pad_id=1)` from `encoder.py` (one line; the rest of the Phase 3-5 code is unchanged).',
         extra_args='--objective mlm --mask-frac 0.3 --seq 2048 --context-length 2048', micro=32)
pretrain('pretrain_hybrid', 'Pretrain 4: Hymba-style hybrid decoder (parallel attention + SSM heads), about 214M parameters, 4B tokens',
         note='Same data, seed, schedule and 2048 context as Pretrain 1; every block runs 8 attention heads (GQA, flash attention) and 8 Mamba-2 SSM heads in parallel and fuses them. See pretrain/hybrid.py.',
         extra_args='--arch hybrid --ffn 2688', micro=16)
