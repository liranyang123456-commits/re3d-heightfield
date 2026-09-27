import os
from pathlib import Path
from huggingface_hub import hf_hub_download

out_dir = Path(r"D:\vggt_omega_checkpoints")
out_dir.mkdir(parents=True, exist_ok=True)
print("Starting download of vggt_omega_1b_416_reproduce.pt...")
p = hf_hub_download(
    repo_id="facebook/VGGT-Omega",
    filename="vggt_omega_1b_416_reproduce.pt",
    local_dir=str(out_dir),
)
print("Download complete! File saved at:", p)
