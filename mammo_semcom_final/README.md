# Mammogram SemCom – final notebooks (01–04)

Run order: 00 (once, builds the cache) -> 01 -> 02 -> 03 -> 04.

## Contents
- 00_data_preparation_10k.ipynb  : builds the cache (images.npy, masks.npy, images.csv, abn.csv) from CBIS-DDSM.zip
- 01_train_localizer_v2.ipynb    : CLEAN_V2 localizer (3 seeds) -> heatmaps_v2.npy, loc_v2.json
- 02_train_classifier_v2.ipynb   : classifier loss comparison -> clf_v2
- 03_semantic_transmission_v2.ipynb : packet-loss transmission, 453 test images
- 04_adversarial_attack_v3.ipynb : adversarial robustness + tag leakage
- src/attack_v2.py, losses.py, train_v2.py, tx_v2.py : NEW modules (put in Drive `mammo_semcom/src`)

## Original modules NOT in this zip (already in your Drive src folder)
config.py, data.py, models.py, train.py, channel.py, pipeline.py, metrics.py, viz.py, __init__.py
-> keep them in /content/drive/MyDrive/mammo_semcom/src together with the 4 files above.

## Drive layout
/content/drive/MyDrive/mammo_semcom/src/            (all .py files)
/content/drive/MyDrive/mammo_semcom_work/           (cache, models, results/)
/content/drive/MyDrive/mammo_semcom_data/CBIS-DDSM.zip

If attack_v2 import fails after replacing the file: restart runtime, or
sys.modules.pop("src.attack_v2", None)
