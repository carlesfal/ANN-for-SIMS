# User Guide — ANN-SIMS Prediction System

## Getting Started

After completing the setup (see [SETUP.md](SETUP.md)), open `http://localhost:3000` in your browser.

---

## 1. Preparing Your Data

Your dataset must be a **CSV or Excel** file with:
- A **header row** as the first row
- One column designated as the **target** (output variable)
- At least one column as a **feature** (input variable)
- Numeric values only (non-numeric rows are dropped)
- No required specific column order

**Example:**
```csv
concentration,temperature,pressure,yield
1.2,300,2.5,0.85
1.5,310,2.8,0.91
...
```

---

## 2. Training a Model

### Step 1: Upload Data
- Go to the **Training** page
- Drag & drop your CSV/Excel file or click to browse
- Maximum file size: 50MB

### Step 2: Configure Training
- **Target Column:** Name of the output variable (e.g., `yield`)
- **Feature Columns:** Leave blank to use all non-target columns, or specify comma-separated names
- **Epochs:** Maximum training iterations (default: 100)
- **Folds:** Number of cross-validation folds (default: 10)
- **Learning Rate:** Adam optimizer step size (default: 0.001)
- **Units / Layers / Dropout:** Neural network architecture parameters
- **Patience:** Early stopping patience (stop if no improvement for N epochs)
- **Hill Pre-training:** Enable to warm-start weights from synthetic Hill-function data

### Step 3: Monitor Training
- Real-time progress bar and percentage
- Live metrics (RMSE, MAE, R², MAPE) per fold
- Training log output
- WebSocket-based live updates

---

## 3. Viewing Results

After training completes, go to the **Results** page:

### Metrics Tab
- **Average metrics** across all folds: RMSE, MAE, R², MAPE with standard deviations
- **Best hyperparameters** found by Keras Tuner
- **Dataset information** (sample count, feature names)

### Per-Fold Tab
- Metrics for each individual cross-validation fold
- The best-performing fold is highlighted

### Charts Tab
- **Predicted vs. Actual** scatter plot across all CV folds
- **Per-Fold RMSE & MAE** bar chart

### Plots Tab
- **3D Surface Plots** for all feature pairs
- **Predicted vs. Actual** scatter plot
- **Residuals** plot
- Click any image to enlarge

### Download Tab
- Download full results as **JSON**
- Download predictions as **CSV**
- Download per-fold metrics as **CSV**

---

## 4. Making Predictions

Go to the **Predictions** page:

### Batch Prediction
1. Enter the **Model Job ID** from a completed training run
2. Upload a new data file (same feature columns as training data)
3. Click **Run Batch Prediction**
4. View predictions with 95% prediction intervals
5. Results show actual values if target column is present in data

### Single Sample Prediction
1. Enter the **Model Job ID**
2. Enter feature values as a JSON object:
   ```json
   {"concentration": 1.5, "temperature": 310, "pressure": 2.8}
   ```
3. Click **Predict**
4. View the prediction with lower and upper 95% PI bounds

---

## 5. Understanding Metrics

| Metric | Description | Better When |
|--------|-------------|-------------|
| **RMSE** | Root Mean Square Error — penalizes large errors | Lower |
| **MAE** | Mean Absolute Error — average absolute deviation | Lower |
| **R²** | Coefficient of Determination — variance explained | Closer to 1.0 |
| **MAPE** | Mean Absolute Percentage Error | Lower |

## 6. Understanding Prediction Intervals

- **95% PI:** The model predicts that the true value falls within [lower_95, upper_95] with 95% confidence
- Computed via **MC Dropout** (if dropout layers present) or residual-based estimation
- Wider intervals indicate higher uncertainty

---

## Tips for Best Results

- Use at least **100 samples** for reliable training (more is better)
- **Normalize** your data beforehand if feature scales differ greatly
- **Enable Hill pre-training** when your data follows sigmoidal/Hill-function patterns
- Use **10 folds** (default) for robust evaluation
- Monitor **R²** as the primary quality metric (>0.9 is excellent)
