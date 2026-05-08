# ANNSIMS Desktop

Artificial Neural Network for SIMS prediction as a **Python desktop application**.

## Features

- Load SIMS datasets from CSV/TSV/TXT/Excel
- Configure preprocessing (normalization + train/val/test split)
- Configure and train a PyTorch ANN model
- Save/load model checkpoints (`.pt`)
- Run predictions and visualize results
- Export prediction results to CSV

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

## Project structure

- `main.py`: app entry point
- `app/`: MVC-style application package
- `tests/`: unit tests
- `ANNSIMS.spec`: PyInstaller spec
- `docs/user_guide.md`: end-user instructions
