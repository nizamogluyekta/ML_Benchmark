# ML Benchmark

Reusable framework for running comparable experiments across multiple ML model families (baseline tree/kernel methods, wavelet hybrids, and neural nets) and datasets.

## Directory Layout
- `configs/` – YAML files describing datasets and benchmark runs.
- `data/` – Raw inputs under `data/raw` and standardized splits produced in `data/processed/<dataset>/`.
- `models/` – Pluggable model implementations grouped by family (baseline, wavelet, neural, deep, keras).
- `reports/` – Per run outputs such as metrics and run markers.
- `scripts/` – Entry points for preprocessing, single-model training, and benchmark orchestration.
- `utils/` – Shared helpers for reporting, label mapping, feature importance, and wavelet transforms.

## Quickstart
1. Define your datasets in `configs/datasets.yaml` (update raw paths, features, and targets).
   - Example: `solar_reference` already points to the copied VeriBilimi splits under `data/raw/solar_reference/`.
2. Adjust `configs/benchmark.yaml` with the datasets/models you want to run.
3. Preprocess data (creates placeholder splits until real logic is added):
   ```bash
   python scripts/preprocess.py --dataset solar_reference --force
   ```
4. Train a single model:
   ```bash
   python scripts/train_model.py --dataset solar_reference --model baseline.random_forest
   ```
5. Run the full benchmark defined in `configs/benchmark.yaml`:
   ```bash
   python scripts/benchmark.py
   ```
   This now writes, for every dataset/model pair, a `predictions.csv`, `metrics.(csv|json)` and a Markdown report under `reports/<dataset>/<model>/`. If `matplotlib`/`seaborn` are installed the report also includes plots (actual vs predicted curves, scatter, error histograms, etc.). An aggregate summary (`reports/benchmark_report.md`) is also generated with cross-model metrics and comparison charts.

## Notes
- Model modules currently expect processed CSVs with numeric columns; extend the preprocessing step to apply real feature engineering.
- Wavelet models now include SVM/RF/XGB/LightGBM/CatBoost plus scikit-learn, Keras, and PyTorch MLP variants; each consumes the shared preprocessing pipeline so comparisons stay fair.
- Baseline models cover SVM, Random Forest, XGBoost, LightGBM, CatBoost, scikit-learn shallow + deep MLPs, and a Keras MLP.
- Optional dependencies (`xgboost`, `lightgbm`, `catboost`, `tensorflow`, `torch`, `matplotlib`, `seaborn`) enable the richer model set and report plots; when missing, the framework logs a clear error for that model and continues with the others.
- Utilities and configs were scaffolded from the `VeriBilimi` project plan and can be iteratively refined as datasets and requirements evolve.
- The legacy VeriBilimi wavelet feature engineering (seasonal/interactions + DWT coefficients) now lives under `utils/wavelet.py` and powers the `models/wavelet/*` implementations.
- Detailed per-model reports in `reports/<dataset>/<model>/` summarize MSE/RMSE/MAE/R², show month-level errors, and list every prediction with its percentage error. When plotting dependencies are unavailable, the Markdown report records that fact so you know to install `matplotlib` + `seaborn` to unlock the figures.
