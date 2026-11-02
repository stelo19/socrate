from huggingface_hub import HfApi
from datasets import get_dataset_config_names, load_dataset

REPO = "allenai/tutormoments-preview"
sha = HfApi().dataset_info(REPO).sha          # exact dataset version
print("revision:", sha)

for cfg in get_dataset_config_names(REPO, revision=sha):
    ds = load_dataset(REPO, cfg, revision=sha)
    for split, d in ds.items():
        print(cfg, split, d.num_rows, d.column_names)
