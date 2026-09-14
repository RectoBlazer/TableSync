"""Keep this project's training caches local and Hub access disabled."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT/'artifacts/hf-cache')
os.environ['HF_DATASETS_CACHE'] = str(ROOT/'artifacts/hf-cache/datasets')
os.environ['TORCH_HOME'] = str(ROOT/'artifacts/torch-cache')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'
