import sys
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path to import scripts/utils if needed
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

app = FastAPI(title="ML Benchmark API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev convenience, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_DIR = ROOT_DIR / "configs"
BENCHMARK_CONFIG = CONFIG_DIR / "benchmark.yaml"
DATASETS_CONFIG = CONFIG_DIR / "datasets.yaml"
MODELS_DIR = ROOT_DIR / "models"

class BenchmarkConfigUpdate(BaseModel):
    datasets: List[str]
    models: List[str]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/config")
def get_config():
    """Get current benchmark configuration."""
    if not BENCHMARK_CONFIG.exists():
        raise HTTPException(status_code=404, detail="Benchmark config not found")
    
    with open(BENCHMARK_CONFIG, "r") as f:
        return yaml.safe_load(f)

@app.post("/config")
def update_config(config_update: BenchmarkConfigUpdate):
    """Update benchmark configuration (datasets and models)."""
    if not BENCHMARK_CONFIG.exists():
        raise HTTPException(status_code=404, detail="Benchmark config not found")
    
    with open(BENCHMARK_CONFIG, "r") as f:
        current_config = yaml.safe_load(f) or {}
    
    # Update only the fields we care about
    current_config["datasets"] = config_update.datasets
    current_config["models"] = config_update.models
    
    with open(BENCHMARK_CONFIG, "w") as f:
        yaml.dump(current_config, f, default_flow_style=False)
    
    return current_config

@app.get("/datasets")
def list_datasets():
    """List all available datasets from datasets.yaml."""
    if not DATASETS_CONFIG.exists():
        # Fallback: list directories in data/processed or data/raw if config missing
        # But for now, let's assume the file exists as per README
        return {"datasets": []}
        
    with open(DATASETS_CONFIG, "r") as f:
        data = yaml.safe_load(f) or {}
        
    # Assuming datasets.yaml structure is a dict where keys are dataset names
    # or a list of dicts. Let's assume keys for now based on typical usage.
    # If it's a list, we'll adjust.
    if isinstance(data, dict):
        return {"datasets": list(data.keys())}
    elif isinstance(data, list):
        # If it's a list of objects with 'name' attribute
        return {"datasets": [d.get("name") for d in data if isinstance(d, dict)]}
    
    return {"datasets": []}

@app.get("/models")
def list_models():
    """List all available models by scanning the models directory."""
    models = []
    
    # Walk through models directory
    for family_dir in MODELS_DIR.iterdir():
        if family_dir.is_dir() and family_dir.name != "__pycache__":
            family = family_dir.name
            for model_file in family_dir.glob("*.py"):
                if model_file.name == "__init__.py":
                    continue
                
                model_name = model_file.stem
                # Construct model ID: family.model_name
                models.append(f"{family}.{model_name}")
                
    return {"models": sorted(models)}

# --- Execution State Management ---

class BenchmarkState:
    def __init__(self):
        self.is_running = False
        self.logs = []
        self.current_task = None

state = BenchmarkState()

import logging
from collections import deque

# Configure logging to capture output
log_capture_string = io.StringIO()
ch = logging.StreamHandler(log_capture_string)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add handler to the root logger or specific loggers
logging.getLogger().addHandler(ch)

# Also capture logs in a deque for the API
class ListHandler(logging.Handler):
    def __init__(self, log_list, max_len=1000):
        super().__init__()
        self.log_list = log_list
        self.max_len = max_len

    def emit(self, record):
        msg = self.format(record)
        self.log_list.append(msg)
        if len(self.log_list) > self.max_len:
            self.log_list.pop(0)

list_handler = ListHandler(state.logs)
list_handler.setFormatter(formatter)
logging.getLogger().addHandler(list_handler)
logging.getLogger("scripts.benchmark").setLevel(logging.INFO)
logging.getLogger("scripts.train_model").setLevel(logging.INFO)


from fastapi import BackgroundTasks
from scripts.benchmark import run_benchmark

def run_benchmark_task():
    state.is_running = True
    state.logs.clear()
    logging.info("Starting benchmark run...")
    try:
        # Ensure reports dir exists
        reports_dir = ROOT_DIR / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        run_benchmark(config_path=BENCHMARK_CONFIG, reports_dir=reports_dir)
        logging.info("Benchmark run completed successfully.")
    except Exception as e:
        logging.error(f"Benchmark run failed: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        state.is_running = False

@app.post("/run")
def trigger_run(background_tasks: BackgroundTasks):
    """Trigger a benchmark run in the background."""
    if state.is_running:
        raise HTTPException(status_code=400, detail="Benchmark is already running")
    
    background_tasks.add_task(run_benchmark_task)
    return {"status": "started", "message": "Benchmark run initiated"}

@app.get("/status")
def get_status():
    """Get the current status of the benchmark run and recent logs."""
    return {
        "is_running": state.is_running,
        "logs": state.logs[-100:] # Return last 100 logs
    }

@app.get("/reports")
def list_reports():
    """List generated reports."""
    reports_dir = ROOT_DIR / "reports"
    if not reports_dir.exists():
        return {"reports": []}
    
    # Look for the summary report
    summary_report = reports_dir / "benchmark_report.md"
    summary_csv = reports_dir / "benchmark_summary.csv"
    
    reports = []
    if summary_report.exists():
        reports.append({
            "name": "Benchmark Summary (Markdown)",
            "path": str(summary_report.relative_to(ROOT_DIR)),
            "type": "markdown"
        })
    if summary_csv.exists():
        reports.append({
            "name": "Benchmark Summary (CSV)",
            "path": str(summary_csv.relative_to(ROOT_DIR)),
            "type": "csv"
        })
        
    # We could also list per-model reports here if needed
    
    return {"reports": reports}

@app.get("/reports/content")
def get_report_content(path: str):
    """Get the content of a specific report file."""
    # Security check: ensure path is within reports dir
    safe_path = (ROOT_DIR / path).resolve()
    if not str(safe_path).startswith(str(ROOT_DIR / "reports")):
         raise HTTPException(status_code=403, detail="Access denied")
    
    if not safe_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    with open(safe_path, "r") as f:
        content = f.read()
        
    return {"content": content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
