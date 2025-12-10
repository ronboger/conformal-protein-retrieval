import logging
import os
import shutil
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)
# Configure root logger for visibility inside Spaces (no-op if already configured)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# Avoid matplotlib cache permission issues in containerized envs
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


def ensure_assets():
    """Download large assets from a Hugging Face Dataset repo if missing.

    Configure via env var HF_DATASET_ID, e.g. "USERNAME/DATASET_NAME".
    This keeps local code paths unchanged by writing into ./data and ./results.
    """
    dataset_id = os.getenv("HF_DATASET_ID")
    if not dataset_id:
        return  # No dataset configured; skip

    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        # If the hub client is not available, skip silently; app can still run
        return

    # Map dataset files -> local target paths expected by the app
    assets: List[Tuple[str, str]] = [
        ("data/lookup_embeddings.npy", "./data/lookup_embeddings.npy"),
        ("data/lookup_embeddings_meta_data.tsv", "./data/lookup_embeddings_meta_data.tsv"),
        ("data/lookup/scope_lookup_embeddings.npy", "./data/lookup/scope_lookup_embeddings.npy"),
        ("data/lookup/scope_lookup.fasta", "./data/lookup/scope_lookup.fasta"),
        ("results/fdr_thresholds.csv", "./results/fdr_thresholds.csv"),
        ("results/fnr_thresholds.csv", "./results/fnr_thresholds.csv"),
        ("results/calibration_probs.csv", "./results/calibration_probs.csv"),
    ]

    for dataset_path, target_path in assets:
        try:
            target_dir = os.path.dirname(target_path)
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            if os.path.exists(target_path):
                continue

            # Use the exact basename to place file at the desired path
            filename = os.path.basename(target_path)
            local_dir = target_dir if target_dir else "."

            # Try to download with subfolder if provided in dataset_path
            subfolder = os.path.dirname(dataset_path).replace("\\", "/")
            subfolder = subfolder if subfolder not in ("", ".") else None

            hf_hub_download(
                repo_id=dataset_id,
                repo_type="dataset",
                filename=filename,
                subfolder=subfolder,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            # Continue on best-effort basis; missing optional files won't block the UI
            logger.warning("Failed to download asset %s from dataset %s: %s", dataset_path, dataset_id, exc)

    # Optional: expand a packed protein_vec_models archive if provided
    # Expecting a tar.gz in the dataset named protein_vec_models.tar.gz
    try:
        pvm_dir = "./protein_vec_models"
        if not os.path.isdir(pvm_dir):
            from huggingface_hub import hf_hub_download
            arc_name = "protein_vec_models.tar.gz"
            arc_dir = "./.cache"
            os.makedirs(arc_dir, exist_ok=True)
            arc_path = hf_hub_download(
                repo_id=dataset_id,
                repo_type="dataset",
                filename=arc_name,
                local_dir=arc_dir,
                local_dir_use_symlinks=False,
            )
            import tarfile
            os.makedirs(pvm_dir, exist_ok=True)
            with tarfile.open(arc_path, "r:gz") as tar:
                def _is_within_directory(directory: str, target: str) -> bool:
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

                for member in tar.getmembers():
                    member_path = os.path.join(pvm_dir, member.name)
                    if not _is_within_directory(pvm_dir, member_path):
                        raise Exception("Tar file contains unsafe path: {}".format(member.name))
                tar.extractall(path=pvm_dir)

            # Flatten single top-level directory if present
            contents = os.listdir(pvm_dir)
            if len(contents) == 1:
                nested = os.path.join(pvm_dir, contents[0])
                if os.path.isdir(nested):
                    for entry in os.listdir(nested):
                        shutil.move(os.path.join(nested, entry), os.path.join(pvm_dir, entry))
                    shutil.rmtree(nested, ignore_errors=True)

            required_files = ("model_protein_moe.py", "utils_search.py")
            missing = [fname for fname in required_files if not os.path.exists(os.path.join(pvm_dir, fname))]
            if missing:
                raise FileNotFoundError(f"Missing expected Protein-Vec assets after extraction: {missing}")
    except Exception as exc:
        # Optional asset; log failures for visibility without crashing the app
        logger.warning("Failed to prepare protein_vec_models assets: %s", exc)


if __name__ == "__main__":
    # Ensure heavy assets exist locally before launching the app
    ensure_assets()

    from protein_conformal.backend.gradio_interface import create_interface

    iface = create_interface()
    iface.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
