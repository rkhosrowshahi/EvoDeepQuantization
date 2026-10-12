"""Download the CIFAR ResNet-18 checkpoints from the Hugging Face Hub on first use.

The checkpoints live at https://huggingface.co/{HF_REPO_ID}. Nothing is
downloaded if the file is already on disk, so a checkpoint you trained yourself
with ``train_cifar.py`` is used as is.
"""

import os
import shutil

HF_REPO_ID = "Raasa/moea-compression-checkpoints"

# (dataset, arch) -> path inside the HF repo.
_HF_CHECKPOINTS = {
    ("cifar10", "resnet18"): "cifar10/resnet18/model_SA_best.pth.tar",
    ("cifar100", "resnet18"): "cifar100/resnet18/model_SA_best.pth.tar",
}
_LOCAL_FILENAME = "0model_SA_best.pth.tar"


def ensure_local_checkpoint(dataset, arch, save_path):
    """Download the (dataset, arch) checkpoint into `save_path` if it is missing."""
    local_path = os.path.join(save_path, _LOCAL_FILENAME)
    if os.path.exists(local_path):
        return

    hf_path = _HF_CHECKPOINTS.get((dataset, arch))
    if hf_path is None:
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download the pretrained checkpoints "
            f"(pip install huggingface_hub, or place the checkpoint at {local_path} yourself)."
        ) from exc

    print(f"Checkpoint not found at {local_path}; downloading {hf_path} "
          f"from https://huggingface.co/{HF_REPO_ID} ...")
    os.makedirs(save_path, exist_ok=True)
    downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=hf_path)
    shutil.copyfile(downloaded, local_path)
    print(f"Saved to {local_path}")
