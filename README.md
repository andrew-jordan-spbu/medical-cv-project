# Medical Image Classification (HAM10000) — Project

This repository contains a ready-to-run project skeleton for a medical image classification task using the HAM10000 dataset.

## Contents
- `data/` - place dataset images and metadata.csv here
- `src/` - (not included in notebook) project code structure
- `models/checkpoints/` - saved model checkpoints
- `notebooks/HAM10000_Colab_Demo.ipynb` - a runnable Colab-ready notebook with demo training and evaluation
- `requirements.txt` - minimal dependencies

## How to run (Colab)
1. Open `notebooks/HAM10000_Colab_Demo.ipynb` in Colab.
2. Upload your `kaggle.json` and run the Kaggle download block (see the notebook).
3. Run cells top-to-bottom. Demo uses a small subset if dataset is missing.

## Dataset
HAM10000 (public):
- Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- ISIC Archive: https://challenge.isic-archive.com/
- Harvard Dataverse DOI: https://doi.org/10.7910/DVN/DBW86T

## Notes
- The notebook includes a *demo mode* which creates a small placeholder dataset using CIFAR images so you can test the pipeline quickly.
- For production training, replace the demo data with the real HAM10000 CSV and images and increase training epochs / batch size.

Citations: Tschandl P., Rosendahl C., Kittler H. The HAM10000 dataset. Sci Data (2018). DOI:10.1038/sdata.2018.161
