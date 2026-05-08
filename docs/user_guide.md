# ANNSIMS Desktop User Guide

## 1. Start the app

```bash
python main.py
```

## 2. Load data

- Go to **Data** tab.
- Click **Load Data** and open a CSV, TSV, TXT, or Excel file.
- Choose feature and target columns.

## 3. Preprocess

- Select normalization (`none`, `min-max`, `z-score`).
- Set train/validation/test split percentages (must sum to 100%).
- Click **Apply Preprocessing**.

## 4. Train model

- Go to **Model** tab.
- Configure hidden layers, activation, optimizer, learning rate, epochs, and batch size.
- Click **Train Model**.
- Save with **Save Model**.

## 5. Predict

- Go to **Predict** tab.
- Click **Load Prediction Input**.
- Click **Run Prediction**.
- Export with **Export Results**.

## 6. Package executable

```bash
pyinstaller ANNSIMS.spec
```

The generated executable will be in `dist/`.
