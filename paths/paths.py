from pathlib import Path

cwd = Path(__file__).resolve().parent
root = cwd.parent
data_dir = root / "data"
images_dir = data_dir / "images"
prompts_file = data_dir / "prompts.txt"
embeddings_dir = root / "embeddings"
results_dir = root / "results"
