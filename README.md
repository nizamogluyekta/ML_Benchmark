# ML Benchmark

Reusable framework for running comparable experiments across multiple ML model families (baseline tree/kernel methods, wavelet hybrids, and neural nets) and datasets.

## Directory Layout
- `configs/` – YAML files describing datasets and benchmark runs.
- `data/` – Raw inputs under `data/raw` and standardized splits produced in `data/processed/<dataset>/`.
- `models/` – Pluggable model implementations grouped by family (baseline, wavelet, neural).
- `reports/` – Per run outputs such as metrics and run markers.
- `scripts/` – Entry points for preprocessing, single-model training, and benchmark orchestration.
- `utils/` – Shared helpers for reporting, label mapping, feature importance, and wavelet transforms.

## Quickstart
1. Define your datasets in `configs/datasets.yaml` (update raw paths, features, and targets).
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

## Notes
- Model modules currently expect processed CSVs with numeric columns; extend the preprocessing step to apply real feature engineering.
- Wavelet models reuse baseline logics for now, keeping a clear hook for custom feature augmentation.
- Utilities and configs were scaffolded from the `VeriBilimi` project plan and can be iteratively refined as datasets and requirements evolve.
