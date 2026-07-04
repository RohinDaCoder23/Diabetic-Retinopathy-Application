# Google Colab quickstart

This project is built to **train on Google Colab** (free GPU). Below is the exact,
copy-paste sequence. Each block is one Colab cell.

> Pick a GPU runtime first: **Runtime → Change runtime type → Hardware accelerator → GPU**.

### 1. Get the code
```python
# Option A: clone your GitHub repo
!git clone https://github.com/<your-username>/diabetic-retinopathy.git
%cd diabetic-retinopathy

# Option B: if you uploaded the folder to Google Drive instead:
# from google.colab import drive; drive.mount('/content/drive')
# %cd /content/drive/MyDrive/diabetic-retinopathy
```

### 2. Install dependencies
```python
!pip install -q -r requirements.txt
```

### 3. Get the APTOS 2019 data (Kaggle API)
```python
# Upload your kaggle.json (Kaggle -> Account -> Create New API Token)
from google.colab import files; files.upload()   # choose kaggle.json
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!pip install -q kaggle
!mkdir -p data/aptos2019
!kaggle competitions download -c aptos2019-blindness-detection -p data/aptos2019
!cd data/aptos2019 && unzip -q aptos2019-blindness-detection.zip
```
Confirm the layout matches `data/README.md` (`data/aptos2019/train.csv` +
`data/aptos2019/train_images/*.png`).

### 4. Confirm the GPU is visible
```python
import torch; print('CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))
```

### 5. SMOKE TEST first (always) — 1 epoch, 2 batches
Catches path/shape/config bugs in ~1 minute before you commit to a long run.
```python
!python src/train.py --config config.yaml --model resnet50 --smoke-test
```
You should see one short epoch run and a checkpoint saved to `models/`.

### 6. Real training
```python
# Transfer-learning models (recommended):
!python src/train.py --config config.yaml --model resnet50
!python src/train.py --config config.yaml --model efficientnet_b0
!python src/train.py --config config.yaml --model densenet121

# The from-scratch baseline:
!python src/train.py --config config.yaml --model custom_cnn
```
Each run saves the **best** checkpoint to `models/<name>_best.pt`, a metrics
history JSON to `reports/`, and a training-curve plot to `reports/figures/`.

### 7. Evaluate + explain + view results
```python
!python src/evaluate.py --config config.yaml --model resnet50      # metrics + confusion matrix
!python src/gradcam.py  --config config.yaml --model resnet50      # Grad-CAM gallery
# then open notebooks/03_results.ipynb to build the comparison table
```

### Tips
- **Save outputs to Drive** so they survive a runtime reset: copy `models/` and
  `reports/` into your mounted Drive after training.
- **Out-of-memory?** Lower `train.batch_size` in `config.yaml` (e.g. 16), or use
  `efficientnet_b0` / keep image size at 224.
- **Reproducibility:** the seed in `config.yaml` is applied everywhere; the same
  config + data reproduces the same results.
