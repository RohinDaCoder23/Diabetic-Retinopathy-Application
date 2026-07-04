# Data directory (gitignored)

This folder is intentionally **empty in git** — datasets are large and are never
committed. This README explains exactly what to download and where to put it so
the code finds it.

## Dataset: APTOS 2019 Blindness Detection

- **Source:** Kaggle competition "APTOS 2019 Blindness Detection".
  Download: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
  (You need a free Kaggle account and must accept the competition rules.)
- **Size:** ~3,662 labeled training fundus photographs.
- **Labels:** a single ordinal grade per image, the ICDR 0–4 scale:

  | grade | meaning              |
  |------:|----------------------|
  | 0     | No DR                |
  | 1     | Mild                 |
  | 2     | Moderate             |
  | 3     | Severe               |
  | 4     | Proliferative DR     |

## Expected layout (drop the download in like this)

The code (and `config.yaml`) expects this exact structure:

```
data/
└── aptos2019/
    ├── train.csv            # columns: id_code, diagnosis
    ├── train_images/        # <id_code>.png  (e.g. 000c1434d8d7.png)
    │   ├── 000c1434d8d7.png
    │   └── ...
    ├── test.csv             # optional (public test set has no labels)
    └── test_images/         # optional
```

`train.csv` looks like:

```csv
id_code,diagnosis
000c1434d8d7,2
001639a390f0,4
0024cdab0c1e,1
...
```

## How to get it onto Google Colab

Two easy options:

**A) Kaggle API (recommended on Colab)**
```bash
# 1) In Colab, upload your kaggle.json (Account -> Create New API Token)
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
pip install kaggle -q

# 2) Download + unzip into the expected folder
mkdir -p data/aptos2019
kaggle competitions download -c aptos2019-blindness-detection -p data/aptos2019
cd data/aptos2019 && unzip -q aptos2019-blindness-detection.zip && cd -
```

**B) Google Drive**
Download the zip once, put it in your Drive, then mount Drive in Colab and copy
it into `data/aptos2019/`.

## Want a different dataset?

The loader is built for the APTOS layout above. EyePACS/2015 and Messidor-2 use
similar `id,label` CSVs — adapt `paths` in `config.yaml` and the column names in
`src/data.py` if you switch.

## Note on images already being .png

APTOS images are PNGs named `<id_code>.png`. If your copy uses a different
extension, update the loader in `src/data.py` (a single line).
