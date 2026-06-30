"""
Backend Gradio interface for Protein Conformal Prediction.

This module provides the Gradio web interface for the protein conformal prediction framework.
It allows users to input protein sequences or FASTA files, generate embeddings,
and perform conformal prediction with statistical guarantees.
"""

import gradio as gr
import json
import numpy as np
import os
import sys
import tempfile
import io
import logging
import shutil
import time
import threading
import inspect
from functools import lru_cache
import pandas as pd
from Bio import SeqIO
from typing import List, Union, Tuple, Dict, Optional, Any, Set

from protein_conformal.util import load_lookup_index, query, read_fasta, get_sims_labels, get_thresh_new_FDR, get_thresh_new, risk, calculate_false_negatives, simplifed_venn_abers_prediction, cap_matches_per_query, sequences_hash, LRUCache, query_count_error


logger = logging.getLogger(__name__)

RESULTS_TABLE_MAX_CHARS = 28
try:
    DATAFRAME_SUPPORTS_MAX_CHARS = "max_chars" in inspect.signature(gr.Dataframe.__init__).parameters
except Exception:
    DATAFRAME_SUPPORTS_MAX_CHARS = False
try:
    DATAFRAME_SUPPORTS_COLUMN_WIDTHS = "column_widths" in inspect.signature(gr.Dataframe.__init__).parameters
except Exception:
    DATAFRAME_SUPPORTS_COLUMN_WIDTHS = False
try:
    DATAFRAME_SUPPORTS_STATIC_COLUMNS = "static_columns" in inspect.signature(gr.Dataframe.__init__).parameters
except Exception:
    DATAFRAME_SUPPORTS_STATIC_COLUMNS = False

# Preferred display widths (in px) for default visible columns:
# Query, UniProt Entry, Protein Name(s), Pfam, Exact Prob, Partial Prob
RESULTS_TABLE_COLUMN_WIDTHS = [280, 120, 260, 180, 115, 115]

# Hard char limits for long display columns (keyed by display header). Truncating
# the data itself guarantees a cell can't blow up the row height when clicked —
# Gradio's max_chars/CSS only affect rendering, not the underlying value, so a
# selected cell would otherwise unfurl the full sequence. Full content stays in
# the detail panel (click a row).
# NOTE: do NOT truncate "Query" — the per-query dropdown matches the full
# query_meta against this column, so a truncated value would break filtering.
RESULTS_DISPLAY_TRUNCATE = {
    "Match Sequence": 30,
    "Match Description": 45,
    "Protein Name(s)": 45,
}


def _truncate_display_columns(display_df, limits=RESULTS_DISPLAY_TRUNCATE):
    """Truncate long string columns in-place (by display header) to `limit` chars + '…'."""
    for col, lim in limits.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda v, _l=lim: (str(v)[:_l] + "…") if isinstance(v, str) and len(v) > _l else v
            )
    return display_df


def _download_from_dataset(path: str) -> str:
    """Download a file from the dataset repo to the given local path."""
    if os.path.exists(path):
        logger.info("Asset already present: %s", path)
        return path

    dataset_id = os.getenv("HF_DATASET_ID")
    if not dataset_id:
        logger.error("Missing %s and HF_DATASET_ID not configured", path)
        raise FileNotFoundError(f"Missing {path} and no HF_DATASET_ID configured.")

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        logger.exception("huggingface_hub import failed while fetching %s", path)
        raise FileNotFoundError(f"Missing {path} and huggingface_hub unavailable: {exc}") from exc

    target_dir = os.path.dirname(path) or "."
    os.makedirs(target_dir, exist_ok=True)

    filename = os.path.basename(path)
    subfolder = os.path.dirname(path).lstrip("./") or None

    try:
        logger.info(
            "Downloading %s (subfolder=%s) from dataset %s to %s",
            filename,
            subfolder,
            dataset_id,
            target_dir,
        )
        downloaded_path = hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=filename,
            subfolder=subfolder,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        logger.info("Download complete: %s", downloaded_path)

        desired_path = os.path.abspath(path)
        actual_path = os.path.abspath(downloaded_path)

        if actual_path != desired_path:
            logger.info("Moving downloaded file from %s to %s", actual_path, desired_path)
            os.makedirs(os.path.dirname(desired_path), exist_ok=True)
            shutil.move(actual_path, desired_path)
        else:
            logger.info("File already at desired location: %s", desired_path)
    except Exception as exc:
        logger.exception("Failed downloading %s from dataset %s", path, dataset_id)
        raise FileNotFoundError(
            f"Unable to download {path} from dataset {dataset_id}: {exc}"
        ) from exc

    return path


def ensure_local_results_file(path: str) -> str:
    """Ensure a small results file exists locally, downloading via dataset if needed."""
    return _download_from_dataset(path)


def ensure_local_data_file(path: str) -> str:
    """Ensure a required data file (lookup embeddings or metadata) exists locally."""
    return _download_from_dataset(path)

def _path_signature(path: str) -> Tuple[str, Optional[float]]:
    resolved = os.path.abspath(path)
    try:
        mtime = os.path.getmtime(resolved)
    except OSError:
        mtime = None
    return resolved, mtime

@lru_cache(maxsize=16)
def _cached_results_dataframe(resolved_path: str, mtime: Optional[float]) -> pd.DataFrame:
    """Load and normalize a CSV, caching by file path + mtime."""
    df = pd.read_csv(resolved_path)

    def _normalize(col: Any) -> str:
        col = str(col)
        col = col.lstrip("\ufeff")  # remove UTF-8 BOM if present
        col = col.strip()
        return col.lower()

    df.rename(columns={col: _normalize(col) for col in df.columns}, inplace=True)

    # Harmonize known aliases
    alias_map = {
        "lambda": "lambda_threshold",
        "lambda threshold": "lambda_threshold",
        "lambda-threshold": "lambda_threshold",
        "threshold_mean": "lambda_threshold",
        "exactfdr": "exact_fdr",
        "partialfdr": "partial_fdr",
        "empirical_fdr_mean": "exact_fdr",
        "exactfnr": "exact_fnr",
        "partialfnr": "partial_fnr",
    }
    for alias, canonical in alias_map.items():
        if alias in df.columns and canonical not in df.columns:
            df.rename(columns={alias: canonical}, inplace=True)

    return df


def load_results_dataframe(path: str, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a results CSV with normalized column names and optional validation."""
    csv_path = ensure_local_results_file(path)
    resolved, mtime = _path_signature(csv_path)
    df = _cached_results_dataframe(resolved, mtime).copy()

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise KeyError(
                f"Columns {missing} not found in {path}. Available columns: {list(df.columns)}"
            )

    return df
import matplotlib.pyplot as plt

DEFAULT_LOOKUP_EMBEDDING = "./data/lookup_embeddings.npy" # default path to the UniProt lookup embeddings 
DEFAULT_LOOKUP_METADATA = "./data/lookup_embeddings_meta_data.tsv" # default path to the UniProt lookup metadata
DEFAULT_CALIBRATION_DATA = "./results/calibration_probs.csv" # default path to the calibration data

DEFAULT_SCOPE_EMBEDDING = "./data/lookup/scope_lookup_embeddings.npy" # default path to the SCOPE lookup embeddings
DEFAULT_SCOPE_METADATA = "./data/lookup/scope_lookup.fasta" # default path to the SCOPE lookup metadata

DEFAULT_AFDB_EMBEDDING = "./data/afdb/afdb_embeddings_protein_vec.npy" # default path to the AFDB lookup embeddings
DEFAULT_AFDB_METADATA = "./data/afdb/afdb_metadata.tsv" # default path to the AFDB lookup metadata

DEFAULT_EUK_EMBEDDING = "./data/euk/euk_embeddings_protein_vec.npy" # default path to the Euk lookup embeddings
DEFAULT_EUK_METADATA = "./data/euk/euk_metadata.tsv" # default path to the Euk lookup metadata

DEFAULT_CLEAN_CENTROID_EMBEDDING = "./data/clean/ec_centroid_embeddings.npy" # CLEAN EC centroid embeddings
DEFAULT_CLEAN_CENTROID_METADATA = "./data/clean/ec_centroid_metadata.tsv" # CLEAN EC centroid metadata
DEFAULT_CLEAN_THRESHOLDS = "./results/clean_thresholds.csv" # CLEAN hierarchical thresholds

# Paths used for temporary storage of uploaded database files
CUSTOM_UPLOAD_EMBEDDING = "./data/custom_uploaded_embedding.npy" # path to the custom uploaded embeddings
CUSTOM_UPLOAD_METADATA = "./data/custom_uploaded_metadata.tsv" # path to the custom uploaded metadata

# Max rows rendered in the browser table per query. The full match set is always
# kept server-side (session/export) so downloads return everything; this only
# bounds the SSE payload sent to the browser.
DISPLAY_ROWS_PER_QUERY = 200

# Max query sequences for the interactive app. Must stay within the GPU embed
# timeout (gpu_embed .get(timeout=...)); genome-scale jobs are routed to the CLI
# (checkpointed, no timeout). Tunable; validated against the A10G embed rate.
MAX_QUERY_SEQUENCES = 5000


def _noop_progress(*args, **kwargs):
    """No-op replacement for gr.Progress.

    Gradio renders progress/status near multiple output components in this UI,
    which made messages like "Embedding sequences..." appear all over the page.
    Keep the internal calls harmless while relying on the Search/Stop button for
    running state.
    """
    return None


# Amino acid validation constants
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')
SPECIAL_CHARS = set('XUB')  # X=unknown, U=selenocysteine, B=ambiguous D/N

# Per-user session data flows through gr.State() and is threaded through the
# search entry points (process_input / process_clean_input) as a `session` dict.
# There is intentionally no module-global session — that would race across
# concurrent users on Modal.


class StageTimer:
    """Utility to measure and log elapsed time for each pipeline stage."""

    def __init__(self):
        self.records: List[Tuple[str, float]] = []

    def track(self, stage_name: str):
        class _StageContext:
            def __init__(self, outer, name):
                self.outer = outer
                self.name = name
                self.start = 0.0

            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                duration = time.perf_counter() - self.start
                self.outer.records.append((self.name, duration))
                logger.info("Stage '%s' finished in %.2fs", self.name, duration)

        return _StageContext(self, stage_name)

    def log_summary(self):
        if not self.records:
            return
        ordered = sorted(self.records, key=lambda item: item[1], reverse=True)
        summary = ", ".join(f"{name}={duration:.2f}s" for name, duration in ordered)
        logger.info("Stage durations (desc): %s", summary)
        slowest_name, slowest_duration = ordered[0]
        logger.info("Slowest stage: %s (%.2fs)", slowest_name, slowest_duration)


LOOKUP_RESOURCE_CACHE: Dict[Tuple[str, Optional[float], str, Optional[float]], Dict[str, Any]] = {}
LOOKUP_RESOURCE_LOCK = threading.Lock()
# Per-key build locks for singleflight index construction (see get_lookup_resources).
LOOKUP_BUILD_LOCKS: Dict[Any, threading.Lock] = {}


def _get_lookup_build_lock(cache_key) -> threading.Lock:
    """Return the (created-on-demand) build lock for a lookup cache key."""
    with LOOKUP_RESOURCE_LOCK:
        lock = LOOKUP_BUILD_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            LOOKUP_BUILD_LOCKS[cache_key] = lock
        return lock

# Process-level embedding cache shared across sessions/users: common queries skip
# the GPU round-trip. Keyed on embedding-mode + sequences_hash. Deterministic, so
# sharing across users is safe (unlike per-user search results).
EMBEDDING_CACHE = LRUCache(256)

# build a cache_key tuple from the absolute path and m


def _persist_uploaded_file(upload_obj: Any, destination: str, preserve_suffix: bool = False) -> str:
    """Save a Gradio upload object (file-like or NamedString) to destination."""
    base_destination = destination
    if preserve_suffix:
        source_name = getattr(upload_obj, "name", None) or (upload_obj if isinstance(upload_obj, str) else None)
        if source_name:
            _, ext = os.path.splitext(source_name)
            if ext:
                destination = os.path.splitext(destination)[0] + ext
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if hasattr(upload_obj, "read"):
        with open(destination, "wb") as dst:
            dst.write(upload_obj.read())
        return destination
    source_path = None
    if hasattr(upload_obj, "name"):
        source_path = upload_obj.name
    elif isinstance(upload_obj, str):
        source_path = upload_obj
    if source_path and os.path.exists(source_path):
        shutil.copy(source_path, destination)
        return destination
    raise AttributeError("Uploaded file object is not readable and has no valid path.")


def _load_lookup_metadata(metadata_path: str) -> Tuple[str, List[str], List[Any]]:
    metadata_path = ensure_local_data_file(metadata_path)
    if metadata_path.lower().endswith(".tsv"):
        df = pd.read_csv(metadata_path, sep="\t")
        if "Sequence" not in df.columns:
            raise KeyError(f"Metadata TSV missing 'Sequence' column: {metadata_path}")
        sequences = df["Sequence"].astype(str).tolist()
        metadata_columns = [col for col in ["Entry", "Pfam", "Protein names"] if col in df.columns]
        if metadata_columns:
            lookup_meta: List[Any] = (
                df[metadata_columns]
                .fillna("")
                .to_dict(orient="records")
            )
        else:
            lookup_meta = [{} for _ in range(len(df))]
        return "tsv", sequences, lookup_meta
    else:
        sequences, lookup_meta = read_fasta(metadata_path)
        return "fasta", sequences, lookup_meta

# When multiple searches hit the same lookup database without those files changing,
# get_lookup_resources returns the preloaded FAISS index + metadata, so we avoid
# repeating the expensive np.load + index build work.
def get_lookup_resources(embedding_path: str, metadata_path: str) -> Dict[str, Any]:
    embedding_path = ensure_local_data_file(embedding_path)
    metadata_path = ensure_local_data_file(metadata_path)
    embedding_sig = _path_signature(embedding_path)
    metadata_sig = _path_signature(metadata_path)
    cache_key = (*embedding_sig, *metadata_sig)

    with LOOKUP_RESOURCE_LOCK:
        cached = LOOKUP_RESOURCE_CACHE.get(cache_key)
        if cached:
            return cached

    # Singleflight: only one thread builds a given DB; concurrent cold-start
    # requests for the same key wait on the per-key lock instead of each loading
    # the ~1 GB embeddings and rebuilding the index.
    build_lock = _get_lookup_build_lock(cache_key)
    with build_lock:
        # Re-check: another thread may have built it while we waited for the lock.
        with LOOKUP_RESOURCE_LOCK:
            cached = LOOKUP_RESOURCE_CACHE.get(cache_key)
            if cached:
                return cached

        metadata_kind, lookup_seqs, lookup_meta = _load_lookup_metadata(metadata_path)
        # Prefer a prebuilt FAISS index (scripts/prebuild_faiss.py). When present, the
        # ~1 GB .npy is never loaded -- the main cold-start cost for big databases.
        lookup_index, num_embeddings = load_lookup_index(embedding_path)

        resource = {
            "index": lookup_index,
            "lookup_seqs": lookup_seqs,
            "lookup_meta": lookup_meta,
            "metadata_kind": metadata_kind,
            "num_embeddings": num_embeddings,
        }

        with LOOKUP_RESOURCE_LOCK:
            LOOKUP_RESOURCE_CACHE[cache_key] = resource

        return resource


def parse_fasta(fasta_content: str) -> Tuple[List[str], List[str]]:
    """ 
    takes in FASTA and returns list of protein sequences and metadata
    
    Returns:
        Tuple of (sequences, metadata) where metadata contains FASTA headers
    """
    sequences = []
    metadata = []
    fasta_file = io.StringIO(fasta_content)
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
        # Store the full header line with '>' prefix to match repo format
        metadata.append(f">{record.id} {record.description}" if record.description != record.id else f">{record.id}")
    return sequences, metadata

def _read_uploaded_text(file_obj) -> Optional[str]:
    """
    Read the raw text of a Gradio File upload, or None if unreadable.

    Supports the different object types Gradio may hand over:
    - file-like objects exposing .read()
    - temporary files with a .name attribute
    - plain filesystem paths (default when type='filepath')
    - dictionaries containing 'path'/'name' metadata
    """
    if file_obj is None:
        return None

    if hasattr(file_obj, "read"):
        data = file_obj.read()
        return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data

    candidate_paths: List[Optional[str]] = []
    if isinstance(file_obj, dict):
        candidate_paths.extend([file_obj.get("path"), file_obj.get("name")])
    else:
        candidate_paths.append(getattr(file_obj, "name", None))
        if isinstance(file_obj, str):
            candidate_paths.append(file_obj)

    for path in candidate_paths:
        if path and os.path.exists(path):
            with open(path, "rb") as fh:
                return fh.read().decode("utf-8", errors="replace")
    return None


def load_uploaded_fasta(file_obj) -> str:
    """Upload handler: show the uploaded FASTA's content in the text box above."""
    return _read_uploaded_text(file_obj) or ""


def process_uploaded_file(file_obj) -> Tuple[List[str], List[str]]:
    """Process an uploaded FASTA file from Gradio's File component into sequences."""
    if file_obj is None:
        return [], []
    fasta_text = _read_uploaded_text(file_obj)
    if not fasta_text:
        raise AttributeError("Uploaded FASTA file is not readable and has no accessible path.")
    return parse_fasta(fasta_text)

def run_embed_protein_vec(
    sequences: List[str],
    progress=None,
    fp16_head: bool = False,
    embedding_device: str = "cpu",
) -> np.ndarray:
    """
    use existing embed_protein_vec.py to generate embeddings

    Args:
        sequences: List of protein sequences
        progress: Gradio progress bar
        fp16_head: EXPERIMENTAL opt-in fp16 Protein-Vec head. Honored by the Modal
            GPU path (gpu_embed monkey-patches this function); ignored by this local
            subprocess fallback.
        embedding_device: cpu, auto, cuda, or mps for the local subprocess.

    Returns:
        NumPy array of embeddings
    """
    if progress is None: progress = _noop_progress
    import subprocess
    import sys
    
    # Create temporary FASTA file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_fasta:
        for i, seq in enumerate(sequences):
            tmp_fasta.write(f">seq_{i}\n{seq}\n")
        tmp_fasta_path = tmp_fasta.name
    
    # Create temporary output file for embeddings
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_out:
        tmp_out_path = tmp_out.name
    
    try:
        progress(0.2, desc="Embedding sequences...")
        
        # Run the embed_protein_vec.py script
        cmd = [
            sys.executable,
            "protein_conformal/embed_protein_vec.py",
            "--input_file", tmp_fasta_path,
            "--output_file", tmp_out_path,
            "--path_to_protein_vec", "protein_vec_models",
            "--device", embedding_device,
        ]

        env = os.environ.copy()
        if embedding_device in {"mps", "auto"}:
            env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".", env=env)
        
        if result.returncode != 0:
            raise Exception(f"embed_protein_vec.py failed: {result.stderr}")
        
        # Load the embeddings
        embeddings = np.load(tmp_out_path)
        
        progress(0.5, desc="Embedding complete. Searching database...")
        return embeddings
        
    finally:
        # Clean up temporary files
        if os.path.exists(tmp_fasta_path):
            os.unlink(tmp_fasta_path)
        if os.path.exists(tmp_out_path):
            os.unlink(tmp_out_path)

def run_embed_clean(sequences: List[str], progress=None) -> np.ndarray:
    """Stub for CLEAN embedding (ESM-1b + LayerNormNet). Monkey-patched on Modal."""
    if progress is None: progress = _noop_progress
    raise NotImplementedError(
        "CLEAN embedding requires GPU. Deploy with Modal or provide pre-computed embeddings."
    )


# CLEAN enzyme search resources cache
CLEAN_RESOURCE_CACHE: Dict[str, Any] = {}
CLEAN_RESOURCE_LOCK = threading.Lock()


def get_clean_resources() -> Dict[str, Any]:
    """Load CLEAN EC centroid embeddings and FAISS L2 index (cached)."""
    import faiss

    with CLEAN_RESOURCE_LOCK:
        if CLEAN_RESOURCE_CACHE:
            return CLEAN_RESOURCE_CACHE

    emb_path = ensure_local_data_file(DEFAULT_CLEAN_CENTROID_EMBEDDING)
    meta_path = ensure_local_data_file(DEFAULT_CLEAN_CENTROID_METADATA)

    centroids = np.load(emb_path).astype(np.float32)
    meta_df = pd.read_csv(meta_path, sep="\t")

    # Build FAISS L2 index
    d = centroids.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(centroids)

    resource = {
        "index": index,
        "centroids": centroids,
        "ec_numbers": meta_df["EC_number"].tolist(),
        "n_proteins": meta_df["n_proteins"].tolist() if "n_proteins" in meta_df.columns else None,
        "num_centroids": centroids.shape[0],
    }

    with CLEAN_RESOURCE_LOCK:
        CLEAN_RESOURCE_CACHE.update(resource)

    return resource


def process_clean_input(
    fasta_text: str,
    upload_file: Optional[Any],
    alpha_value: float,
    session: Optional[dict] = None,
    progress=None,
    clean_embed_fn=None,
) -> Tuple[str, pd.DataFrame, dict]:
    """Process input for CLEAN enzyme classification mode.

    Returns ``(summary_json, display_df, session)`` where ``session`` is the
    per-user state dict (carries results for export); it is threaded back into
    ``gr.State`` by ``on_submit`` rather than stored in a module global.
    """
    if progress is None: progress = _noop_progress
    import json

    stage_timer = StageTimer()

    try:
        # Step 1: Parse FASTA
        with stage_timer.track("parse_input"):
            sequences = []
            query_meta = []
            if upload_file is not None:
                sequences, query_meta = process_uploaded_file(upload_file)
            elif fasta_text and fasta_text.strip():
                sequences, query_meta = parse_fasta(fasta_text)
            else:
                return json.dumps({"error": "No FASTA input provided."}, indent=2), pd.DataFrame(), (session or {})

        if not sequences:
            return json.dumps({"error": "No sequences found in FASTA input."}, indent=2), pd.DataFrame(), (session or {})

        # Step 2: Embed with CLEAN (ESM-1b + LayerNormNet)
        progress(0.1, desc="Embedding with ESM-1b + CLEAN...")
        with stage_timer.track("clean_embedding"):
            embed_clean = clean_embed_fn or run_embed_clean
            embeddings = embed_clean(sequences, progress)

        # Step 3: Load CLEAN resources and search
        progress(0.5, desc="Searching EC centroid database...")
        with stage_timer.track("clean_search"):
            resources = get_clean_resources()
            index = resources["index"]
            ec_numbers = resources["ec_numbers"]
            n_proteins_list = resources["n_proteins"]
            k = min(50, resources["num_centroids"])

            D, I = index.search(embeddings, k)  # D = squared L2 distances, I = indices
            D = np.sqrt(D)  # Convert to regular L2 to match calibration thresholds

        # Step 4: Look up hierarchical threshold
        progress(0.7, desc="Applying hierarchical conformal threshold...")
        with stage_timer.track("threshold_lookup"):
            threshold = None
            try:
                thresh_df = load_results_dataframe(
                    DEFAULT_CLEAN_THRESHOLDS,
                    required_columns=["alpha", "lambda_threshold"],
                )
                closest_idx = (thresh_df["alpha"] - alpha_value).abs().idxmin()
                threshold = float(thresh_df.iloc[closest_idx]["lambda_threshold"])
                actual_alpha = float(thresh_df.iloc[closest_idx]["alpha"])
                test_loss = float(thresh_df.iloc[closest_idx].get("test_loss_mean", alpha_value))
            except (FileNotFoundError, KeyError) as e:
                logger.warning(f"CLEAN thresholds not available: {e}. Returning top-k results.")
                threshold = None
                actual_alpha = alpha_value
                test_loss = None

        # Step 5: Build results
        progress(0.8, desc="Building results...")
        with stage_timer.track("results_packaging"):
            results = []
            for i, (distances, indices) in enumerate(zip(D, I)):
                for dist, idx in zip(distances, indices):
                    if threshold is not None and dist > threshold:
                        continue
                    result = {
                        "query_meta": query_meta[i] if i < len(query_meta) else f"seq_{i}",
                        "ec_number": ec_numbers[idx],
                        "distance": float(dist),
                    }
                    if n_proteins_list is not None:
                        result["n_proteins_in_ec"] = n_proteins_list[idx]
                    results.append(result)

            results_df = pd.DataFrame(results) if results else pd.DataFrame()

            summary = {
                "status": "success",
                "mode": "Enzyme Classification (CLEAN)",
                "matches_found": len(results),
                "hierarchical_guarantee": {
                    "alpha": actual_alpha,
                    "threshold": threshold,
                    "meaning": f"Expected max hierarchical loss <= {actual_alpha:.1f}",
                    "empirical_test_loss": test_loss,
                },
                "loss_level_guide": {
                    "0": "Exact EC match",
                    "1": "Same sub-subclass (4th digit differs)",
                    "2": "Same subclass (3rd digit differs)",
                    "3": "Same class (2nd digit differs)",
                    "4": "Different class",
                },
            }
            if threshold is None:
                summary["warning"] = "No threshold table found. Showing top-k results without conformal guarantee."

            # Format display DataFrame
            if not results_df.empty:
                display_header_map = {
                    "query_meta": "Query",
                    "ec_number": "Predicted EC",
                    "distance": "L2 Distance",
                    "n_proteins_in_ec": "# Proteins in EC",
                }
                TRUNCATE_CLEAN = {"query_meta": 50}
                display_df = results_df.copy()
                for col, lim in TRUNCATE_CLEAN.items():
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(
                            lambda v, _l=lim: (str(v)[:_l] + "\u2026") if isinstance(v, str) and len(v) > _l else v
                        )
                display_df = display_df.rename(columns=display_header_map)
            else:
                display_df = pd.DataFrame()

            # Per-user session for export (returned to on_submit -> gr.State)
            new_session = {
                "results": {"summary": summary, "matches": results, "threshold": threshold},
                "parameters": {"mode": "clean", "alpha": alpha_value},
            }

        progress(1.0, desc="Enzyme classification complete!")
        return json.dumps(summary, indent=2, default=str), display_df, new_session

    except Exception as e:
        return json.dumps({"error": f"Error during CLEAN search: {str(e)}"}, indent=2), pd.DataFrame(), (session or {})
    finally:
        stage_timer.log_summary()


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Validate a protein sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not sequence:
        return False, "Sequence is empty"
    
    sequence = sequence.strip().upper()
    invalid_chars = set(sequence) - VALID_AA - SPECIAL_CHARS
    
    if invalid_chars:
        return False, f"Invalid amino acid characters: {', '.join(invalid_chars)}"
    
    # Check if the sequence contains too many special characters (>10%)
    special_char_count = sum(1 for aa in sequence if aa in SPECIAL_CHARS)
    if special_char_count / len(sequence) > 0.1:
        return False, f"Too many special characters ({special_char_count})"
    
    return True, ""

def highlight_sequence(sequence: str) -> str:
    """
    Add HTML highlighting to a protein sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        HTML-formatted sequence with highlighting
    """
    sequence = sequence.upper()
    highlighted = []
    
    # Color scheme for different amino acid types
    # Hydrophobic: red, Polar: blue, Charged: green, Special: grey
    color_map = {
        'A': 'red', 'V': 'red', 'L': 'red', 'I': 'red', 'M': 'red', 'F': 'red', 'W': 'red', 'P': 'red',  # Hydrophobic
        'S': 'blue', 'T': 'blue', 'N': 'blue', 'Q': 'blue', 'Y': 'blue', 'C': 'blue',  # Polar
        'D': 'green', 'E': 'green', 'K': 'green', 'R': 'green', 'H': 'green',  # Charged
        'G': 'purple',  # Glycine (special)
        'X': 'grey', 'U': 'grey', 'B': 'grey'  # Special
    }
    
    for aa in sequence:
        if aa in color_map:
            highlighted.append(f'<span style="color:{color_map[aa]}">{aa}</span>')
        else:
            highlighted.append(aa)
    
    return ''.join(highlighted)

def run_search(query_embeddings: np.ndarray,
               query_seqs: List[str],
               query_meta: List[str],
               lookup_embedding_path: str = DEFAULT_LOOKUP_EMBEDDING,
               lookup_metadata_path: str = DEFAULT_LOOKUP_METADATA,
               threshold: float = 0.0,
               k: int = 1000,
               progress=None) -> pd.DataFrame:
    """
    Run protein search with a specified similarity threshold.
    
    Args:
        query_embeddings: Query embeddings
        query_seqs: Query sequences
        query_meta: Query metadata (FASTA headers)
        lookup_embedding_path: Path to lookup embeddings
        lookup_metadata_path: Path to lookup metadata
        threshold: Similarity threshold from conformal prediction
        k: Maximum number of neighbors
        progress: Gradio progress bar
        
    Returns:
        DataFrame with search results matching repository format
    """
    if progress is None: progress = _noop_progress
    progress(0.1, desc="Loading database...")
    resources = get_lookup_resources(lookup_embedding_path, lookup_metadata_path)
    lookup_database = resources["index"]
    lookup_seqs = resources["lookup_seqs"]
    lookup_meta = resources["lookup_meta"]
    metadata_kind = resources["metadata_kind"]
    max_neighbors = min(k, resources["num_embeddings"])
    
    progress(0.7, desc="Searching database...")
    D, I = query(lookup_database, query_embeddings, max_neighbors)
    
    # Create results DataFrame matching repository output format
    results = []
    for i, (indices, distances) in enumerate(zip(I, D)):
        for idx, distance in zip(indices, distances):
            if distance >= threshold:  # Only include results that meet the threshold
                result = {
                    "query_seq": query_seqs[i],
                    "query_meta": query_meta[i],
                    "lookup_seq": lookup_seqs[idx],
                    "D_score": distance,
                }
                if metadata_kind == "tsv":
                    meta_row = lookup_meta[idx]
                    result["lookup_entry"] = meta_row.get("Entry", "")
                    result["lookup_pfam"] = meta_row.get("Pfam", "")
                    result["lookup_protein_names"] = meta_row.get("Protein names", "")
                else:
                    result["lookup_meta"] = lookup_meta[idx]
                results.append(result)
    results = pd.DataFrame(results)
    
    progress(1.0, desc="Search complete!")
    return results


"""
Below are code for I/O and generating gradio website
"""
def process_input(input_text: str,
                  fasta_text: str,
                  upload_file: Optional[Any],
                  input_type: str,
                  risk_type: str,
                  risk_value: float,
                  max_results: int,
                  use_protein_vec: bool,
                  custom_embeddings: Optional[Any] = None,
                  lookup_db: str = DEFAULT_LOOKUP_EMBEDDING,
                  metadata_db: str = DEFAULT_LOOKUP_METADATA,
                  custom_lookup_upload: Optional[Any] = None,
                  custom_metadata_upload: Optional[Any] = None,
                  match_type: str = "Exact",
                  min_probability: float = 0.5,
                  fp16_head: bool = False,
                  embedding_device: str = "cpu",
                  session: Optional[dict] = None,
                  progress=None,
                  embed_fn=None) -> Tuple[str, pd.DataFrame, dict]:
    """Wrapper that instruments the main pipeline with timing information.

    ``session`` carries this user's prior state (used for embedding/FAISS cache
    reuse); the updated session is returned and stored in ``gr.State`` by
    ``on_submit``.
    """
    if progress is None: progress = _noop_progress
    stage_timer = StageTimer()
    t_start = time.perf_counter()
    try:
        summary, df, session_out = _process_input_impl(
            stage_timer,
            input_text,
            fasta_text,
            upload_file,
            input_type,
            risk_type,
            risk_value,
            max_results,
            use_protein_vec,
            custom_embeddings,
            lookup_db,
            metadata_db,
            custom_lookup_upload,
            custom_metadata_upload,
            match_type,
            min_probability,
            fp16_head,
            embedding_device,
            session,
            progress,
            embed_fn,
        )
        # Surface total search wall-time in the summary the user sees.
        if isinstance(summary, dict):
            summary["search_time_seconds"] = round(time.perf_counter() - t_start, 2)
        return json.dumps(summary, indent=2, default=str), df, session_out
    finally:
        stage_timer.log_summary()


def _process_input_impl(stage_timer: StageTimer,
                        input_text: str,
                        fasta_text: str,
                        upload_file: Optional[Any],
                        input_type: str,
                        risk_type: str,
                        risk_value: float,
                        max_results: int,
                        use_protein_vec: bool,
                        custom_embeddings: Optional[Any] = None,
                        lookup_db: str = DEFAULT_LOOKUP_EMBEDDING,
                        metadata_db: str = DEFAULT_LOOKUP_METADATA,
                        custom_lookup_upload: Optional[Any] = None,
                        custom_metadata_upload: Optional[Any] = None,
                        match_type: str = "Exact",
                        min_probability: float = 0.5,
                        fp16_head: bool = False,
                        embedding_device: str = "cpu",
                        session: Optional[dict] = None,
                        progress=None,
                        embed_fn=None) -> Tuple[Dict[str, Any], pd.DataFrame, dict]:
    """
    Process the input and generate predictions.
    
    Args:
        input_text: Text input containing sequences (deprecated, kept for compatibility)
        fasta_text: Text input containing FASTA formatted sequences 
        upload_file: Uploaded FASTA file
        input_type: Type of input (always "fasta_format" now)
        risk_type: Type of risk to control (FDR or FNR)
        risk_value: Risk tolerance value (0.01 - 0.2)
        use_protein_vec: Whether to use Protein-Vec for embeddings
        custom_embeddings: User-uploaded embeddings file
        lookup_db: Path to pre-embedded lookup database
        metadata_db: Path to lookup metadata
        custom_lookup_upload: User-uploaded lookup database file
        custom_metadata_upload: User-uploaded metadata file
        progress: Gradio progress bar
        
    Returns:
        Tuple containing:
        - Summary information (for the summary JSON display)
        - Table data (for the DataFrame display)
        - Complete results (for the raw JSON output)
    """
    if progress is None: progress = _noop_progress

    # Ensure risk_value is numeric (Dropdown may return a string)
    risk_value = float(risk_value)

    # Step 1: Get sequences and metadata from FASTA input
    with stage_timer.track("parse_input"):
        sequences = []
        query_meta = []
        if upload_file is not None:
            sequences, query_meta = process_uploaded_file(upload_file)
        elif fasta_text and fasta_text.strip():
            sequences, query_meta = parse_fasta(fasta_text)
        else:
            return {"error": "No FASTA input provided. Please enter FASTA content or upload a FASTA file."}, pd.DataFrame(), (session or {})

    if not sequences and custom_embeddings is None:
        return {"error": "No sequences found in the FASTA input. Please check your input format."}, pd.DataFrame(), (session or {})

    too_many = query_count_error(len(sequences), MAX_QUERY_SEQUENCES)
    if too_many:
        return {"error": too_many}, pd.DataFrame(), (session or {})

    # Ensure conformal_results is initialized
    conformal_results = {}

    # Handle custom uploaded database files if present
    if custom_lookup_upload is not None:
        try:
            lookup_db = _persist_uploaded_file(custom_lookup_upload, CUSTOM_UPLOAD_EMBEDDING)
        except Exception as e:
            return {"error": f"Error processing custom database: {str(e)}"}, pd.DataFrame(), (session or {})

    if custom_metadata_upload is not None:
        try:
            metadata_db = _persist_uploaded_file(custom_metadata_upload, CUSTOM_UPLOAD_METADATA, preserve_suffix=True)
        except Exception as e:
            return {"error": f"Error processing custom metadata: {str(e)}"}, pd.DataFrame(), (session or {})

    # ---- Caching: reuse embeddings and FAISS results across parameter changes ----
    input_hash = sequences_hash(sequences)
    # fp16 head produces (slightly) different embeddings, so give it a distinct hash
    # across ALL cache layers (session cache, EMBEDDING_CACHE, and the FAISS result
    # cache key all derive from input_hash) -- else toggling it would serve stale
    # fp32-head embeddings.
    if fp16_head:
        input_hash = f"{input_hash}:fp16head"
    if embedding_device and embedding_device != "cpu":
        input_hash = f"{input_hash}:device:{embedding_device}"
    cached = (session or {}).get("_cache", {})
    new_cache = cached  # preserved unless a fresh search overwrites it below
    embeddings = None

    # Step 2: Get embeddings (skip if same input sequences are cached)
    if cached.get("input_hash") == input_hash and cached.get("embeddings") is not None:
        embeddings = cached["embeddings"]
        sequences = cached["sequences"]
        query_meta = cached["query_meta"]
        logger.info("Reusing cached embeddings for %d sequences", len(sequences))
    elif use_protein_vec and not custom_embeddings:
        try:
            emb_key = f"protein_vec:{input_hash}"
            embeddings = EMBEDDING_CACHE.get(emb_key)
            if embeddings is not None:
                logger.info("Process-level embedding cache hit for %d sequences", len(sequences))
            else:
                with stage_timer.track("protein_vec_embedding"):
                    embed = embed_fn or run_embed_protein_vec
                    # Modal passes gpu_embed(sequences, progress=None, fp16_head=False)
                    # explicitly into create_interface(). The local subprocess fallback
                    # accepts embedding_device for developer-only CPU/MPS/CUDA testing.
                    if embedding_device and embedding_device != "cpu" and embed is run_embed_protein_vec:
                        embeddings = embed(
                            sequences,
                            progress,
                            fp16_head=fp16_head,
                            embedding_device=embedding_device,
                        )
                    else:
                        embeddings = embed(
                            sequences,
                            progress,
                            fp16_head=fp16_head,
                        )
                EMBEDDING_CACHE.put(emb_key, embeddings)
        except Exception as e:
            return {"error": f"Error generating embeddings: {str(e)}"}, pd.DataFrame(), (session or {})
    elif custom_embeddings:
        try:
            with stage_timer.track("load_custom_embeddings"):
                with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
                    tmp.write(custom_embeddings.read())
                    tmp_path = tmp.name
                embeddings = np.load(tmp_path)
                os.unlink(tmp_path)
        except Exception as e:
            return {"error": f"Error loading embeddings: {str(e)}"}, pd.DataFrame(), (session or {})
    else:
        return {"error": "Either Protein-Vec must be enabled or custom embeddings must be provided"}, pd.DataFrame(), (session or {})
    
    # Step 3: Perform conformal prediction (or probability filter)
    try:
        # Determine which database is being used
        database_type = "Custom"
        if lookup_db == DEFAULT_LOOKUP_EMBEDDING:
            database_type = "Swiss-Prot"
        elif lookup_db == DEFAULT_SCOPE_EMBEDDING:
            database_type = "SCOPE"
        elif lookup_db == DEFAULT_AFDB_EMBEDDING:
            database_type = "AFDB (Clustered)"
        elif lookup_db == DEFAULT_EUK_EMBEDDING:
            database_type = "Euk (74K)"

        is_partial = match_type.lower() == "partial"
        match_type_label = "partial" if is_partial else "exact"
        is_prob_filter = risk_type == "Probability Filter"

        if is_prob_filter:
            # Probability Filter mode — no conformal threshold, filter by p0 after search
            threshold = 0.0
            closest_alpha = None
            empirical_risk = None
            conformal_results = {
                "threshold": 0.0,
                "risk_type": "probability_filter",
                "match_type": match_type_label,
                "empirical_risk": None,
                "has_probability_calibration": True,
            }
        else:
            with stage_timer.track("threshold_lookup"):
                if risk_type.lower() == "fdr":
                    fdr_file = "./results/fdr_thresholds_partial.csv" if is_partial else "./results/fdr_thresholds.csv"
                    try:
                        threshold_df = load_results_dataframe(
                            fdr_file,
                            required_columns=["alpha", "lambda_threshold"],
                        )
                    except (FileNotFoundError, KeyError) as e:
                        if is_partial:
                            logger.warning(f"Partial FDR thresholds not available, using exact: {e}")
                            threshold_df = load_results_dataframe(
                                "./results/fdr_thresholds.csv",
                                required_columns=["alpha", "lambda_threshold"],
                            )
                            match_type_label = "exact (partial unavailable)"
                        else:
                            raise
                    closest_idx = (threshold_df['alpha'] - risk_value).abs().idxmin()
                    closest_alpha = threshold_df.iloc[closest_idx]['alpha']
                    threshold = threshold_df.iloc[closest_idx]['lambda_threshold']
                    fdr_col = 'partial_fdr' if is_partial else 'exact_fdr'
                    empirical_risk = threshold_df.iloc[closest_idx].get(fdr_col, None)
                    if abs(closest_alpha - risk_value) > 0.001:
                        logger.warning(f"Requested alpha={risk_value} not available. Using closest alpha={closest_alpha}")
                else:
                    fnr_file = "./results/fnr_thresholds_partial.csv" if is_partial else "./results/fnr_thresholds.csv"
                    try:
                        threshold_df = load_results_dataframe(
                            fnr_file,
                            required_columns=["alpha", "lambda_threshold"],
                        )
                    except (FileNotFoundError, KeyError) as e:
                        if is_partial:
                            logger.warning(f"Partial FNR thresholds not available, using exact: {e}")
                            threshold_df = load_results_dataframe(
                                "./results/fnr_thresholds.csv",
                                required_columns=["alpha", "lambda_threshold"],
                            )
                            match_type_label = "exact (partial unavailable)"
                        else:
                            raise
                    closest_idx = (threshold_df['alpha'] - risk_value).abs().idxmin()
                    closest_alpha = threshold_df.iloc[closest_idx]['alpha']
                    threshold = threshold_df.iloc[closest_idx]['lambda_threshold']
                    fnr_col = 'partial_fnr' if is_partial else 'exact_fnr'
                    empirical_risk = threshold_df.iloc[closest_idx].get(fnr_col, None)
                    if abs(closest_alpha - risk_value) > 0.001:
                        logger.warning(f"Requested alpha={risk_value} not available. Using closest alpha={closest_alpha}")

            conformal_results = {
                "threshold": float(threshold),
                "risk_type": risk_type.lower(),
                "match_type": match_type_label,
                "empirical_risk": float(empirical_risk) if empirical_risk is not None else None,
                "has_probability_calibration": True,
            }

        # Step 4: Run FAISS search (cached across parameter changes for same input + db + k)
        search_key = (input_hash, lookup_db, metadata_db, max_results)
        if cached.get("search_key") == search_key and cached.get("raw_matches") is not None:
            raw_matches = cached["raw_matches"]
            logger.info("Reusing cached FAISS results (%d raw matches)", len(raw_matches))
        else:
            with stage_timer.track("database_search"):
                results_df = run_search(
                    embeddings,
                    sequences,
                    query_meta,
                    lookup_db,
                    metadata_db,
                    threshold=0.0,  # Get ALL k results; filter by threshold below
                    k=max_results,
                    progress=progress
                )
            raw_matches = results_df.to_dict(orient="records")
            # Calibrate raw matches (instant: np.interp lookup table)
            try:
                cal_df = load_results_dataframe(
                    DEFAULT_CALIBRATION_DATA,
                    required_columns=["similarity", "prob_exact_p0", "prob_exact_p1"],
                )
                cal_df = cal_df.sort_values('similarity')
                _sim_cal = cal_df['similarity'].values
                _p0_exact_cal = cal_df['prob_exact_p0'].values
                _p1_exact_cal = cal_df['prob_exact_p1'].values
                _has_partial = 'prob_partial_p0' in cal_df.columns and 'prob_partial_p1' in cal_df.columns
                if _has_partial:
                    _p0_partial_cal = cal_df['prob_partial_p0'].values
                    _p1_partial_cal = cal_df['prob_partial_p1'].values
                for m in raw_matches:
                    sim = m["D_score"]
                    m["p0"] = float(np.interp(sim, _sim_cal, _p0_exact_cal))
                    m["p1"] = float(np.interp(sim, _sim_cal, _p1_exact_cal))
                    if _has_partial:
                        m["p0_partial"] = float(np.interp(sim, _sim_cal, _p0_partial_cal))
                        m["p1_partial"] = float(np.interp(sim, _sim_cal, _p1_partial_cal))
            except Exception as e:
                logger.warning(f"Raw match calibration failed: {e}")
                # Flag calibration failure so UI can warn user
                for m in raw_matches:
                    m["_calibration_failed"] = True
            # Update cache
            new_cache = {
                "input_hash": input_hash,
                "embeddings": embeddings,
                "sequences": sequences,
                "query_meta": query_meta,
                "search_key": search_key,
                "raw_matches": raw_matches,
            }

        # Apply conformal threshold to cached raw results
        sim_threshold = conformal_results.get("threshold", 0.0)
        all_matches = [m for m in raw_matches if m["D_score"] >= sim_threshold]

        # Format probability display strings (p0/p1 already computed on raw_matches)
        with stage_timer.track("probability_formatting"):
            for i, match in enumerate(all_matches):
                p0_e = match.get("p0", 0)
                p1_e = match.get("p1", 0)
                mean_e = (p0_e + p1_e) / 2
                half_e = abs(p1_e - p0_e) / 2
                all_matches[i]["prob_exact"] = f"{mean_e*100:.1f}% \u00b1 {half_e*100:.1f}%"
                p0_p = match.get("p0_partial")
                if p0_p is not None:
                    p1_p = match.get("p1_partial", 0)
                    mean_p = (p0_p + p1_p) / 2
                    half_p = abs(p1_p - p0_p) / 2
                    all_matches[i]["prob_partial"] = f"{mean_p*100:.1f}% \u00b1 {half_p*100:.1f}%"

        # For Probability Filter mode, post-filter by p0 (conservative lower bound)
        if is_prob_filter:
            p0_key = "p0_partial" if is_partial else "p0"
            all_matches = [m for m in all_matches if m.get(p0_key, 0) >= min_probability]

        total_matches = len(all_matches)
        # Only the table display is capped; all_matches (full set) still flows into
        # the session for download/export below.
        display_matches = cap_matches_per_query(all_matches, "query_meta", DISPLAY_ROWS_PER_QUERY)
        results_df = pd.DataFrame(display_matches) if display_matches else pd.DataFrame()

        with stage_timer.track("results_packaging"):
            # Build summary
            summary = {
                "status": "success",
                "matches_found": total_matches,
            }
            if is_prob_filter:
                summary["search_mode"] = {
                    "type": "Probability Filter",
                    "min_probability_p0": min_probability,
                    "match_type": match_type_label,
                }
            else:
                summary["risk_control"] = {
                    "type": conformal_results["risk_type"].upper(),
                    "alpha_used": float(closest_alpha),
                    "alpha_requested": risk_value,
                    "threshold": round(conformal_results["threshold"], 10),
                    "empirical_risk": round(conformal_results["empirical_risk"], 4) if conformal_results["empirical_risk"] else None,
                }
            summary["search_config"] = {
                "database": database_type,
                "match_type": conformal_results["match_type"],
                "max_k": max_results,
            }
            # Table is capped per query; tell the user the download has the rest.
            if len(display_matches) < total_matches:
                summary["display_note"] = (
                    f"Showing {len(display_matches)} of {total_matches} matches "
                    f"(top {DISPLAY_ROWS_PER_QUERY} per query). Use Download to get all."
                )
            # Threshold >= 1.0 warning
            if not is_prob_filter and conformal_results["threshold"] >= 1.0:
                summary["warning"] = "Threshold >= 1.0: no results can pass at this alpha level. Try a less strict (higher) alpha."

            # Calibration failure warning
            if all_matches and all_matches[0].get("_calibration_failed"):
                summary["calibration_warning"] = "Probability calibration unavailable. Probability columns show 0.000 and should be ignored."

            # Experimental fp16 head note (opt-in via Advanced Options)
            if fp16_head:
                summary["fp16_head_note"] = (
                    "EXPERIMENTAL fp16 Protein-Vec head ON: faster embedding, but may "
                    "alter borderline matches (conformal guarantee not exact)."
                )

            # Remove None values
            summary = {k: v for k, v in summary.items() if v is not None}

            # Complete results for export
            complete_results = {
                "summary": summary,
                "matches": all_matches,
                "threshold": conformal_results["threshold"],
                "risk_type": conformal_results["risk_type"],
                "match_type": conformal_results["match_type"],
                "empirical_risk": conformal_results["empirical_risk"]
            }

            # Build this user's session (cache preserved across parameter changes)
            new_session = {
                "results": complete_results,
                "parameters": {
                    "risk_type": risk_type,
                    "risk_value": risk_value,
                    "max_results": max_results,
                    "database_type": database_type,
                },
                "_cache": new_cache,
            }

        # Format display DataFrame
        # Internal columns hidden from display: D_score, p0, p1, p0_partial, p1_partial, query_seq
        display_header_map = {
            "query_meta": "Query",
            "query_seq": "Query Sequence",
            "lookup_seq": "Match Sequence",
            "lookup_meta": "Match Description",
            "lookup_entry": "UniProt Entry",
            "lookup_pfam": "Pfam",
            "lookup_protein_names": "Protein Name(s)",
            "prob_exact": "Exact Prob",
            "prob_partial": "Partial Prob",
        }
        preferred_order = [
            "query_meta",
            "lookup_entry",
            "lookup_protein_names",
            "lookup_pfam",
            "prob_exact",
            "prob_partial",
        ]
        HIDDEN_COLS = {"D_score", "p0", "p1", "p0_partial", "p1_partial", "query_seq", "lookup_seq", "lookup_meta"}
        if not results_df.empty:
            display_columns = [col for col in preferred_order if col in results_df.columns
                               and col not in HIDDEN_COLS]
            display_columns.extend([col for col in results_df.columns if col not in display_columns
                                    and col not in HIDDEN_COLS])
            display_df = results_df.reindex(columns=display_columns).copy()
            display_df = display_df.rename(columns=display_header_map)
            _truncate_display_columns(display_df)
        else:
            display_df = pd.DataFrame()

        return summary, display_df, new_session
    except Exception as e:
        error_message = {"error": f"Error during search: {str(e)}"}
        return error_message, pd.DataFrame(), (session or {})

def export_current_results(format_type: str, session: dict) -> Tuple[str, Optional[str]]:
    """
    Export the current results in the specified format.
    All matches (not just displayed ones) will be included in the export.
    
    Args:
        format_type: Format to export (csv, json)
        
    Returns:
        Tuple of (status message string, file path for download)
    """
    if not session or "results" not in session:
        return "❌ No results to export. Run a search first.", None

    try:
        # Create a directory for exported reports if it doesn't exist
        os.makedirs("exported_reports", exist_ok=True)

        # Create a unique filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        risk_type = (session.get("parameters", {}).get("risk_type") or "risk").lower()
        threshold = session.get("results", {}).get("threshold")
        threshold_tag = "thr_unknown"
        if isinstance(threshold, (int, float)):
            threshold_tag = f"thr_{threshold:.4f}".replace(".", "p")
        file_path = os.path.join(
            "exported_reports",
            f"results_{timestamp}_{risk_type}_{threshold_tag}.{format_type}",
        )

        # Export the results
        if format_type == "csv":
            if "matches" in session["results"]:
                # Export ALL matches, not just the displayed ones
                df = pd.DataFrame(session["results"]["matches"])
                df.to_csv(file_path, index=False)
                total_exported = len(df)
            else:
                return "❌ No matches to export.", None
        elif format_type == "json":
            with open(file_path, 'w') as f:
                # For JSON export, we include the full result structure
                json.dump(session["results"], f, indent=2, default=str)
                total_exported = len(session["results"].get("matches", []))
        else:
            return f"❌ Unsupported format: {format_type}", None

        import os as _os
        fname = _os.path.basename(file_path)
        return f"✅ Exported {total_exported:,} records to `{fname}`", file_path

    except Exception as e:
        return f"❌ Error exporting results: {e}", None

def export_embeddings(session: dict) -> Tuple[str, Optional[str]]:
    """Export cached query embeddings as a .npy file for reuse with the CLI."""
    cache = (session or {}).get("_cache", {})
    embeddings = cache.get("embeddings")
    query_meta = cache.get("query_meta")
    if embeddings is None:
        return "❌ No embeddings available. Run a search first.", None

    try:
        os.makedirs("exported_reports", exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_seqs, dim = embeddings.shape

        npy_path = os.path.join("exported_reports", f"query_embeddings_{timestamp}.npy")
        np.save(npy_path, embeddings)

        import os as _os
        fname = _os.path.basename(npy_path)
        msg = f"✅ Saved {n_seqs} embeddings ({dim}-d) to `{fname}`"
        msg += " — order matches input FASTA. Use with: `cpr search --query <this file>`"
        return msg, npy_path
    except Exception as e:
        return f"❌ Error saving embeddings: {e}", None


def _format_summary_html(summary_json_str, elapsed_s=None):
    """Turn a search-summary JSON string into a human-readable HTML block
    with inline styles (immune to theme/CSS overrides).

    Handles both Protein Search and CLEAN (Enzyme Classification) summaries,
    plus error dicts.
    """
    _STYLE = ("background:#f7f5fb;color:#2d2440;padding:8px 14px;"
              "border-radius:8px;border:1px solid #e6ddf5;font-size:14px;line-height:1.6;")
    _EXTRA_STYLE = "font-size:13px;margin-top:6px;"

    try:
        s = json.loads(summary_json_str) if isinstance(summary_json_str, str) else summary_json_str
    except (json.JSONDecodeError, TypeError):
        s = summary_json_str if isinstance(summary_json_str, dict) else {}

    def _esc(t):
        return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    strong_style = 'color:#2d2440 !important;font-weight:700;'

    # --- Error case ---
    if isinstance(s, dict) and "error" in s:
        return f'<div style="{_STYLE}"><strong style="{strong_style}">❌ Search failed</strong> — {_esc(s["error"])}</div>'

    if not isinstance(s, dict):
        return f'<div style="{_STYLE}">{_esc(s)}</div>' if s else ""

    matches = s.get("matches_found", 0)
    elapsed = elapsed_s if elapsed_s is not None else s.get("search_time_seconds")
    time_str = f" · {elapsed:.1f}s" if elapsed else ""

    # --- CLEAN (Enzyme Classification) ---
    if s.get("mode") == "Enzyme Classification (CLEAN)":
        hg = s.get("hierarchical_guarantee") or {}
        alpha = hg.get("alpha")
        meaning = hg.get("meaning", "")
        line = f"<strong style=\"{strong_style}\">✅ {matches} EC prediction{'s' if matches != 1 else ''}</strong>"
        detail_parts = []
        if alpha is not None:
            detail_parts.append(f"α={alpha}")
        if meaning:
            detail_parts.append(meaning.replace("Expected max hierarchical loss ", "loss ≤ ").replace("<=", "≤"))
        if detail_parts:
            line += " · " + " · ".join(detail_parts) + time_str
        body = line
        if s.get("warning"):
            body += f'<div style="{_EXTRA_STYLE}">⚠️ {_esc(s["warning"])}</div>'
        return f'<div style="{_STYLE}">{body}</div>'

    # --- Protein Search ---
    line = f"<strong style=\"{strong_style}\">✅ {matches:,} match{'es' if matches != 1 else ''}</strong>"

    detail_parts = []
    rc = s.get("risk_control")
    sm = s.get("search_mode")
    if rc:
        rtype = rc.get("type", "")
        alpha_used = rc.get("alpha_used")
        if alpha_used is not None:
            detail_parts.append(f"{rtype} α={alpha_used}")
        else:
            detail_parts.append(f"{rtype}")
    elif sm:
        min_p = sm.get("min_probability_p0")
        if min_p is not None:
            detail_parts.append(f"Probability ≥ {min_p:.2f}")
        detail_parts.append(f"{sm.get('match_type', '')} match")

    sc = s.get("search_config") or {}
    db = sc.get("database", "")
    mtype = sc.get("match_type", "")
    max_k = sc.get("max_k")
    if db:
        detail_parts.append(db)
    if mtype and not sm:
        detail_parts.append(f"{mtype} match")
    if max_k:
        detail_parts.append(f"top {max_k:,} searched")

    if detail_parts:
        line += " · " + " · ".join(detail_parts) + time_str

    body = line
    for key, icon in [("display_note", "ℹ️"), ("warning", "⚠️"),
                       ("calibration_warning", "⚠️"), ("fp16_head_note", "⚡")]:
        val = s.get(key)
        if val:
            body += f'<div style="{_EXTRA_STYLE}">{icon} {_esc(val)}</div>'
    return f'<div style="{_STYLE}">{body}</div>'


def _format_export_html(status_str):
    """Turn an export status string into a one-line HTML message with inline styles."""
    _STYLE = ("background:#f7f5fb;color:#2d2440;padding:8px 14px;"
              "border-radius:8px;border:1px solid #e6ddf5;font-size:14px;")

    def _esc(t):
        return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    if not status_str:
        return ""
    return f'<div style="{_STYLE}">{_esc(status_str)}</div>'


def _setup_status_html():
    """Render a compact readiness checklist for local search assets."""
    import importlib.util

    def _size(path):
        if not os.path.exists(path):
            return "missing"
        size = os.path.getsize(path)
        if size >= 1024**3:
            return f"{size / 1024**3:.1f} GB"
        if size >= 1024**2:
            return f"{size / 1024**2:.0f} MB"
        return f"{size / 1024:.0f} KB"

    def _row(name, ok, detail):
        color = "#1f7a3f" if ok else "#9a3412"
        bg = "#eef7ee" if ok else "#fff6e5"
        status = "ready" if ok else "needs setup"
        return (
            f'<div style="display:flex;justify-content:space-between;gap:12px;'
            f'padding:8px 10px;border-radius:8px;background:{bg};margin:6px 0;">'
            f'<span style="color:#2d2440;font-weight:600;">{name}</span>'
            f'<span style="color:{color};">{status} · {detail}</span></div>'
        )

    lookup = DEFAULT_LOOKUP_EMBEDDING
    metadata = DEFAULT_LOOKUP_METADATA
    faiss_index = lookup[:-4] + ".faissindex" if lookup.endswith(".npy") else lookup + ".faissindex"
    model_files = [
        "protein_vec_models/protein_vec.ckpt",
        "protein_vec_models/protein_vec_params.json",
        "protein_vec_models/model_protein_moe.py",
        "protein_vec_models/utils_search.py",
        "protein_vec_models/model_protein_vec_single_variable.py",
        "protein_vec_models/embed_structure_model.py",
    ]
    deps = ["sentencepiece", "google.protobuf", "pytorch_lightning", "h5py"]

    rows = [
        _row("Swiss-Prot embeddings", os.path.exists(lookup), _size(lookup)),
        _row("Swiss-Prot metadata", os.path.exists(metadata), _size(metadata)),
        _row("Prebuilt FAISS index", os.path.exists(faiss_index), _size(faiss_index)),
        _row(
            "Protein-Vec model files",
            all(os.path.exists(p) for p in model_files),
            f"{sum(os.path.exists(p) for p in model_files)}/{len(model_files)} files",
        ),
        _row(
            "Python dependencies",
            all(importlib.util.find_spec(d) is not None for d in deps),
            ", ".join(d.split(".")[0] for d in deps),
        ),
    ]
    return (
        '<div style="font-size:0.93em;color:#2d2440;">'
        '<div style="margin-bottom:6px;color:#555;">Local readiness for Protein Search.</div>'
        + "".join(rows) +
        '</div>'
    )


def create_interface(embed_fn=None, clean_embed_fn=None):
    """
    Create and configure the Gradio interface for protein conformal prediction.

    Args:
        embed_fn: Optional Protein-Vec embedding callable. Public Modal deploys
            pass a GPU-backed function here; local runs use run_embed_protein_vec.
        clean_embed_fn: Optional CLEAN embedding callable. Modal deploys pass a
            GPU-backed function here; local runs use run_embed_clean.

    Returns:
        Gradio interface object
    """
    embed_fn = embed_fn or run_embed_protein_vec
    clean_embed_fn = clean_embed_fn or run_embed_clean
    # Custom CSS for better styling
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .gradio-container {
        font-family: 'Inter', sans-serif !important;
    }

    /* Section headers (### headings) — calm left-border accent, no gradient.
       The brand purple gradient is reserved for the hero banner only. */
    .prose h3, .md h3 {
        color: #2d2440 !important;
        border-left: 4px solid #764ba2;
        padding: 4px 0 4px 12px;
        margin: 16px 0 12px 0;
        font-size: 1.15em;
        font-weight: 600;
        background: transparent !important;
    }

    /* Keep inline dataframe editor compact: no row blow-up on long cell values. */
    #results-table textarea,
    #results-table [contenteditable="true"] {
        white-space: nowrap !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        line-height: 1.35 !important;
        min-height: 2.1em !important;
        max-height: 2.1em !important;
        height: 2.1em !important;
    }

    /* De-brand: footer_links=[] removes the main Gradio footer at launch;
       these rules catch any residual "Built with Gradio" chrome or API link. */
    footer { display: none !important; }
    a[href*="gradio.app"], a[href*="gradio.cc"] { display: none !important; }

    /* Force dark text on the HTML wrapper containers themselves — Gradio's
       theme can set color:white on the wrapper, making inline-styled content
       invisible. This overrides the wrapper, and inline styles handle the rest. */
    #search-summary, #export-status, #results-count {
        color: #2d2440 !important;
        background: transparent !important;
    }
    #search-summary div, #export-status div, #results-count div {
        color: #2d2440 !important;
    }

    /* Monospace font for FASTA input — sequences are more readable in mono. */
    #fasta-input textarea {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 13px !important;
    }

    /* Single, compact home for the search progress overlay. */
    #search-status {
        margin-top: 8px !important;
        min-height: 48px !important;
        background: #f7f5fb !important;
        border: 1px solid #e6ddf5 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    #search-status, #search-status * {
        color: #2d2440 !important;
    }

    """

    # Brand theme: purple/violet accents on the Soft base so every control
    # (primary buttons, sliders, focus rings, selected tabs) matches the hero
    # gradient instead of defaulting to Gradio's stock blue.
    brand_theme = gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="violet",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    )

    with gr.Blocks(title="Conformal Protein Retrieval", analytics_enabled=False) as interface:
        # Per-user session state — threaded through process_input/process_clean_input
        # so concurrent users never share results (no module-global session).
        session_state = gr.State({})

        # Header — compact hero. The purple gradient lives here only.
        gr.HTML("""
        <div style="text-align: center; padding: 14px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 14px;">
            <h1 style="color: white; font-size: 1.8em; margin: 0 0 4px 0;">Conformal Protein Retrieval</h1>
            <p style="color: rgba(255,255,255,0.92); font-size: 1em; margin: 0;">
                Functional protein mining with statistical guarantees
                &nbsp;·&nbsp;
                <a href="https://www.nature.com/articles/s41467-024-55676-y" target="_blank" style="color: rgba(255,255,255,0.85); text-decoration: underline;">Boger et al., Nature Communications 2025</a>
            </p>
        </div>
        """)

        # "How it works" — collapsible so it doesn't push inputs below the fold.
        with gr.Accordion("How it works", open=False):
            gr.Markdown(
                "Enter protein sequences in **FASTA** format, choose **FDR** or **FNR** risk control, "
                "and retrieve functionally similar proteins from Swiss-Prot (540K), AFDB (clustered "
                "AlphaFold DB), or a custom database — with provable error-rate guarantees."
            )

        # Main interface with tabs
        with gr.Tabs():
            with gr.TabItem("Search"):
                with gr.Row():
                    # Left column - Input
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Protein Sequences")

                        fasta_text = gr.TextArea(
                            lines=8,
                            label="FASTA Content",
                            elem_id="fasta-input",
                            value=""">sp|Q99ZW2|CAS9_STRP1 CRISPR-associated endonuclease Cas9 OS=Streptococcus pyogenes serotype M1 GN=cas9
MDKKYSIGLDIGTNSVGWAVITDEYKVPSKKFKVLGNTDRHSIKKNLIGALLFDSGETAEATRLKRTARRRYTRRKNRICYLQEIFSNEMAKVDDSFFHRLEESFLVEEDKKHERHPIFGNIVDEVAYHEKYPTIYHLRKKLVDSTDKADLRLIYLALAHMIKFRGHFLIEGDLNPDNSDVDKLFIQLVQTYNQLFEENPINASGVDAKAILSARLSKSRRLENLIAQLPGEKKNGLFGNLIALSLGLTPNFKSNFDLAEDAKLQLSKDTYDDDLDNLLAQIGDQYADLFLAAKNLSDAILLSDILRVNTEITKAPLSASMIKRYDEHHQDLTLLKALVRQQLPEKYKEIFFDQSKNGYAGYIDGGASQEEFYKFIKPILEKMDGTEELLVKLNREDLLRKQRTFDNGSIPHQIHLGELHAILRRQEDFYPFLKDNREKIEKILTFRIPYYVGPLARGNSRFAWMTRKSEETITPWNFEEVVDKGASAQSFIERMTNFDKNLPNEKVLPKHSLLYEYFTVYNELTKVKYVTEGMRKPAFLSGEQKKAIVDLLFKTNRKVTVKQLKEDYFKKIECFDSVEISGVEDRFNASLGTYHDLLKIIKDKDFLDNEENEDILEDIVLTLTLFEDREMIEERLKTYAHLFDDKVMKQLKRRRYTGWGRLSRKLINGIRDKQSGKTILDFLKSDGFANRNFMQLIHDDSLTFKEDIQKAQVSGQGDSLHEHIANLAGSPAIKKGILQTVKVVDELVKVMGRHKPENIVIEMARENQTTQKGQKNSRERMKRIEEGIKELGSQILKEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDHIVPQSFLKDDSIDNKVLTRSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFDNLTKAERGGLSELDKAGFIKRQLVETRQITKHVAQILDSRMNTKYDENDKLIREVKVITLKSKLVSDFRKDFQFYKVREINNYHHAHDAYLNAVVGTALIKKYPKLESEFVYGDYKVYDVRKMIAKSEQEIGKATAKYFFYSNIMNFFKTEITLANGEIRKRPLIETNGETGEIVWDKGRDFATVRKVLSMPQVNIVKKTEVQTGGFSKESILPKRNSDKLIARKKDWDPKKYGGFDSPTVAYSVLVVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAKGYKEVKKDLIIKLPKYSLFELENGRKRMLASAGELQKGNELALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVEQHKHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAENIIHLFTLTNLGAPAAFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDLSQLGGD""",
                        )

                        upload_file = gr.File(
                            label="Or Upload FASTA File",
                            file_types=[".fasta", ".fa", ".txt"]
                        )

                        # Example buttons
                        gr.Markdown("**Quick Examples:**")
                        with gr.Row():
                            example_btn_cas9 = gr.Button("Cas9", size="sm", elem_id="example-cas9")
                            example_btn_reca = gr.Button("RecA", size="sm", elem_id="example-reca")
                            example_btn_cox1 = gr.Button("CoxA", size="sm", elem_id="example-coxa")
                            example_btn_insulin = gr.Button("Insulin", size="sm", elem_id="example-insulin")
                            example_btn_syn30 = gr.Button("Syn3.0", size="sm", elem_id="example-syn30")

                        use_protein_vec = gr.State(value=True)
                        custom_embeddings_state = gr.State(value=None)

                    # Right column - Parameters
                    with gr.Column(scale=1):
                        gr.Markdown("### Search Parameters")

                        # Top-level mode selector
                        analysis_mode = gr.Radio(
                            ["Protein Search (Protein-Vec)", "Enzyme Classification (CLEAN)"],
                            label="Analysis Mode",
                            value="Protein Search (Protein-Vec)",
                            info="Protein Search: Find similar proteins | Enzyme Classification: Predict EC numbers"
                        )

                        # --- Protein Search parameters (visible by default) ---
                        with gr.Group(visible=True) as protein_search_params:
                            risk_type = gr.Radio(
                                ["FDR", "FNR", "Probability Filter"],
                                label="Search Mode",
                                value="FDR",
                                info="FDR/FNR: Conformal guarantees | Probability Filter: Direct threshold on match probability"
                            )

                            FDR_ALPHAS = ["0.005", "0.01", "0.02", "0.05", "0.1", "0.15", "0.2"]
                            FNR_ALPHAS = ["0.005", "0.01", "0.02", "0.05", "0.1", "0.15", "0.2"]

                            risk_value = gr.Dropdown(
                                choices=FDR_ALPHAS,
                                value="0.1",
                                label="Risk Level (α)",
                                info="Lower = stricter threshold, fewer but more confident results"
                            )

                            min_probability = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.01,
                                label="Minimum Match Probability (p0)",
                                info="Conservative lower-bound probability for filtering matches",
                                visible=False,
                            )

                            match_type = gr.Radio(
                                ["Exact", "Partial"],
                                label="Pfam Match Type",
                                value="Exact",
                                info="Exact: All Pfam domains match | Partial: At least one Pfam domain overlaps"
                            )

                            hide_uncharacterized = gr.Checkbox(
                                label="Hide uncharacterized proteins",
                                value=False,
                                info="Filter out results with 'Uncharacterized' in protein name"
                            )

                        # --- Enzyme Classification parameters (hidden by default) ---
                        with gr.Group(visible=False) as clean_params:
                            CLEAN_ALPHAS = ["0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]

                            clean_alpha = gr.Dropdown(
                                choices=CLEAN_ALPHAS,
                                value="1.0",
                                label="Max Hierarchical Loss (α)",
                                info="0.5=near-exact EC, 1.0=same sub-subclass, 2.0=same subclass, 3.0=same class"
                            )

                            gr.Markdown("""
                            **EC Hierarchy**: `class.subclass.sub-subclass.serial` (4 levels)
                            - α=0.5: Near-exact EC match (loss ≈ 0.5)
                            - α=1.0: Same sub-subclass — 4th digit may differ (loss ≤ 1)
                            - α=2.0: Same subclass — 3rd digit may differ (loss ≤ 2)
                            - α=3.0: Same class — 2nd digit may differ (loss ≤ 3)
                            """)

                        # Database options in accordion (Protein Search only)
                        with gr.Accordion("Advanced Options", open=False, visible=True) as advanced_options:
                            db_type = gr.Radio(
                                ["Swiss-Prot (540K)", "SCOPE", "AFDB (Clustered)", "Euk (74K)", "Custom"],
                                label="Database",
                                value="Swiss-Prot (540K)",
                                info="Select lookup database",
                                elem_id="database-radio",
                            )

                            max_results_slider = gr.Slider(
                                minimum=1,
                                maximum=5000,
                                value=1000,
                                step=100,
                                label="Max Results per Query (k)",
                                info="Maximum neighbors per query"
                            )

                            custom_lookup_upload = gr.File(
                                label="Custom Embeddings (.npy)",
                                file_types=[".npy"],
                                visible=False,
                            )

                            custom_metadata_upload = gr.File(
                                label="Custom Metadata (.fasta/.tsv)",
                                file_types=[".fasta", ".fa", ".tsv"],
                                visible=False,
                            )

                            fp16_head_checkbox = gr.Checkbox(
                                label="Experimental: fp16 Protein-Vec head",
                                value=False,
                                info="Faster embedding, but may alter borderline matches "
                                     "(conformal guarantee not exact). GPU only.",
                            )

                            # Developer-only local device selector. Keep hidden/commented
                            # for the public web UI, which uses Modal GPU embedding via
                            # modal_app.py's gpu_embed monkey-patch.
                            # embedding_device_dropdown = gr.Dropdown(
                            #     ["CPU", "Auto", "Apple MPS (experimental)", "CUDA"],
                            #     value="CPU",
                            #     label="Embedding Device",
                            #     info=(
                            #         "Local-only experimental embedding device. "
                            #         "Public Modal deployments use GPU automatically."
                            #     ),
                            # )

                            with gr.Accordion("Setup readiness", open=False):
                                gr.HTML(_setup_status_html())

                        lookup_db_state = gr.State(value=DEFAULT_LOOKUP_EMBEDDING)
                        metadata_db_state = gr.State(value=DEFAULT_LOOKUP_METADATA)

                        with gr.Row():
                            submit_btn = gr.Button("Search", variant="primary", size="lg", scale=3)
                            stop_btn = gr.Button("⏹ Stop", variant="stop", size="lg", visible=False, scale=3)
                            clear_btn = gr.Button("Clear", size="sm", variant="secondary", scale=1)
                        search_status = gr.HTML("", visible=False, elem_id="search-status")
                        gr.HTML(
                            '<div style="color:#666;font-size:0.88em;margin-top:6px;line-height:1.4;">'
                            'Repeat searches are cached: changing FDR/FNR or match type reuses embeddings and database hits when possible. '
                            'Changing the FASTA input or database reruns the expensive steps.'
                            '</div>'
                        )

                # Results section
                gr.Markdown("### Results")

                with gr.Accordion("How to interpret results", open=False):
                    gr.Markdown("""
                    - **Exact Prob** estimates the probability that the matched protein has the same Pfam domain set as the query.
                    - **Partial Prob** estimates the probability that at least one Pfam domain overlaps.
                    - **FDR α** controls the expected fraction of false discoveries among returned hits.
                    - **FNR α** favors recall by controlling missed true matches.
                    - The table is capped for readability; **Export Results** includes the full match set.
                    """)

                with gr.Row():
                    query_filter = gr.Dropdown(
                        choices=["All queries"],
                        value="All queries",
                        label="Filter by Query",
                        interactive=True,
                        scale=3,
                    )

                results_count_md = gr.HTML("", elem_id="results-count")

                # Empty-state placeholder — shown until the first search completes.
                empty_state = gr.HTML(
                    '<div style="text-align:center;padding:40px 20px;color:#999;font-size:1.1em;">'
                    'Run a search to see matched proteins</div>',
                    visible=True,
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        results_table_kwargs = {
                            "label": "Matches",
                            "wrap": False,
                            "interactive": True,
                            "elem_id": "results-table",
                            "show_row_numbers": True,
                        }
                        if DATAFRAME_SUPPORTS_MAX_CHARS:
                            results_table_kwargs["max_chars"] = RESULTS_TABLE_MAX_CHARS
                        if DATAFRAME_SUPPORTS_COLUMN_WIDTHS:
                            results_table_kwargs["column_widths"] = RESULTS_TABLE_COLUMN_WIDTHS
                        if DATAFRAME_SUPPORTS_STATIC_COLUMNS:
                            # Keep selection/click behavior while disabling in-cell editing.
                            results_table_kwargs["static_columns"] = list(range(32))
                        results_table = gr.Dataframe(**results_table_kwargs)

                    with gr.Column(scale=1):
                        sequence_detail = gr.Code(
                            label="Selected Match (click to copy)",
                            language=None,
                            interactive=False,
                            visible=False,
                        )

                        results_summary = gr.HTML("", elem_id="search-summary")

                        with gr.Row():
                            export_format = gr.Radio(
                                ["csv", "json"],
                                label="Export Format",
                                value="csv",
                                scale=2
                            )
                            export_btn = gr.Button("Export Results", size="sm", scale=1)
                            save_emb_btn = gr.Button("Save Embeddings (.npy)", size="sm", scale=1)

                        export_status = gr.HTML("", elem_id="export-status")
                        export_download = gr.File(label="Download", interactive=False)

                with gr.Accordion("Diagnostics: probability vs rank", open=False):
                    prob_plot = gr.Plot(
                        label="Match Probability vs. Ordered Hit Rank (1 = top hit)",
                        visible=False,
                    )

            with gr.TabItem("About"):
                gr.HTML("""
                <div style="max-width:960px;margin:0 auto;color:#2d2440;">
                  <h2 style="margin-bottom:6px;">Conformal Protein Retrieval</h2>
                  <p style="font-size:1.05em;color:#555;margin-top:0;">
                    Functional protein mining with calibrated probabilities and conformal error-rate guarantees.
                  </p>

                  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:18px 0;">
                    <div style="background:#f7f5fb;border:1px solid #e6ddf5;border-radius:10px;padding:16px;">
                      <h3 style="margin-top:0;color:#2d2440;">Protein Search</h3>
                      <p>Find functionally similar proteins from Swiss-Prot, SCOPE, AFDB, Eukaryotic, or custom databases.</p>
                      <p><strong>Controls:</strong> FDR, FNR, or direct probability filtering.</p>
                    </div>
                    <div style="background:#f7f5fb;border:1px solid #e6ddf5;border-radius:10px;padding:16px;">
                      <h3 style="margin-top:0;color:#2d2440;">Enzyme Classification</h3>
                      <p>Predict EC numbers using CLEAN-style embeddings and hierarchical conformal guarantees.</p>
                      <p><strong>Controls:</strong> maximum hierarchical loss α.</p>
                    </div>
                  </div>

                  <div style="background:white;border:1px solid #eee;border-radius:10px;padding:16px;margin:18px 0;">
                    <h3 style="margin-top:0;color:#2d2440;">Pipeline</h3>
                    <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;color:#555;">
                      <span style="background:#f7f5fb;padding:8px 10px;border-radius:8px;">FASTA</span>
                      <span>→</span>
                      <span style="background:#f7f5fb;padding:8px 10px;border-radius:8px;">Embedding</span>
                      <span>→</span>
                      <span style="background:#f7f5fb;padding:8px 10px;border-radius:8px;">FAISS search</span>
                      <span>→</span>
                      <span style="background:#f7f5fb;padding:8px 10px;border-radius:8px;">Conformal filtering</span>
                      <span>→</span>
                      <span style="background:#f7f5fb;padding:8px 10px;border-radius:8px;">Ranked results</span>
                    </div>
                  </div>

                  <div style="display:flex;gap:10px;flex-wrap:wrap;margin:18px 0;">
                    <div style="background:#eef7ee;border:1px solid #cfe8cf;border-radius:999px;padding:8px 12px;">Syn3.0: 59/149 genes annotated</div>
                    <div style="background:#eef3ff;border:1px solid #d3dcff;border-radius:999px;padding:8px 12px;">Swiss-Prot: 540K lookup proteins</div>
                    <div style="background:#fff6e5;border:1px solid #ffe1a8;border-radius:999px;padding:8px 12px;">Protein-Vec: 512-d embeddings</div>
                  </div>
                </div>
                """)
                gr.Markdown("""
                ### Citation

                ```bibtex
                @article{boger2025functional,
                  title={Functional protein mining with conformal guarantees},
                  author={Boger, Ron S and Chithrananda, Seyone and Angelopoulos, Anastasios N
                          and Yoon, Peter H and Jordan, Michael I and Doudna, Jennifer A},
                  journal={Nature Communications},
                  volume={16},
                  pages={85},
                  year={2025}
                }
                ```
                """)

        # Example sequences
        EXAMPLE_CAS9 = """>sp|Q99ZW2|CAS9_STRP1 CRISPR-associated endonuclease Cas9 OS=Streptococcus pyogenes serotype M1 GN=cas9
MDKKYSIGLDIGTNSVGWAVITDEYKVPSKKFKVLGNTDRHSIKKNLIGALLFDSGETAEATRLKRTARRRYTRRKNRICYLQEIFSNEMAKVDDSFFHRLEESFLVEEDKKHERHPIFGNIVDEVAYHEKYPTIYHLRKKLVDSTDKADLRLIYLALAHMIKFRGHFLIEGDLNPDNSDVDKLFIQLVQTYNQLFEENPINASGVDAKAILSARLSKSRRLENLIAQLPGEKKNGLFGNLIALSLGLTPNFKSNFDLAEDAKLQLSKDTYDDDLDNLLAQIGDQYADLFLAAKNLSDAILLSDILRVNTEITKAPLSASMIKRYDEHHQDLTLLKALVRQQLPEKYKEIFFDQSKNGYAGYIDGGASQEEFYKFIKPILEKMDGTEELLVKLNREDLLRKQRTFDNGSIPHQIHLGELHAILRRQEDFYPFLKDNREKIEKILTFRIPYYVGPLARGNSRFAWMTRKSEETITPWNFEEVVDKGASAQSFIERMTNFDKNLPNEKVLPKHSLLYEYFTVYNELTKVKYVTEGMRKPAFLSGEQKKAIVDLLFKTNRKVTVKQLKEDYFKKIECFDSVEISGVEDRFNASLGTYHDLLKIIKDKDFLDNEENEDILEDIVLTLTLFEDREMIEERLKTYAHLFDDKVMKQLKRRRYTGWGRLSRKLINGIRDKQSGKTILDFLKSDGFANRNFMQLIHDDSLTFKEDIQKAQVSGQGDSLHEHIANLAGSPAIKKGILQTVKVVDELVKVMGRHKPENIVIEMARENQTTQKGQKNSRERMKRIEEGIKELGSQILKEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDHIVPQSFLKDDSIDNKVLTRSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFDNLTKAERGGLSELDKAGFIKRQLVETRQITKHVAQILDSRMNTKYDENDKLIREVKVITLKSKLVSDFRKDFQFYKVREINNYHHAHDAYLNAVVGTALIKKYPKLESEFVYGDYKVYDVRKMIAKSEQEIGKATAKYFFYSNIMNFFKTEITLANGEIRKRPLIETNGETGEIVWDKGRDFATVRKVLSMPQVNIVKKTEVQTGGFSKESILPKRNSDKLIARKKDWDPKKYGGFDSPTVAYSVLVVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAKGYKEVKKDLIIKLPKYSLFELENGRKRMLASAGELQKGNELALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVEQHKHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAENIIHLFTLTNLGAPAAFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDLSQLGGD"""

        EXAMPLE_RECA = """>sp|P0A7G6|RECA_ECOLI Protein RecA OS=Escherichia coli (strain K12) GN=recA
MAIDENKQKALAAALGQIEKQFGKGSIMRLGEDRSMDVETISTGSLSLDIALGAGGLPMGRIVEIYGPESSGKTTLTLQVIAAAQREGKTCAFIDAEHALDPIYARKLGVDIDNLLCSQPDTGEQALEICDALARSGAVDVIVVDSVAALTPKAEIEGEIGDSHMGLAARMMSQAMRKLAGNLKQSNTLLIFINQIRMKIGVMFGNPETTTGGNALKFYASVRLDIRRIGAVKEGENVVGSETRVKVVKNKIAAPFKQAEFQILYGEGINFYGELVDLGVKEKLIEKAGAWYSYKGEKIGQGKANATAWLKDNPETAKEIEKKVRELLLSNPNSTPDFSVDDSEGVAETNEDF"""

        EXAMPLE_COX1 = """>sp|P0AFC7|COX1_ECOLI Cytochrome c oxidase subunit I OS=Escherichia coli (strain K12) GN=coxA
MFQLMPLDLIILLAACAGVSFGTKYENVGSIYSAFPLMIAGFNPSGPIILAVAGLGLTAISSLLRDLPNRLSTLPVGGYGVVLGLLGAGISGAAVDMAAARTAYTIVGNAGGVSYISPPPNGTPVINVVQIMTDTANRRQLKKKNQLGHAFSPQLLGRSLALMGTSFGDVWMTAVGAGMAYFSTSGLAMGITTGLSVSFGGGDPEFMARYLLVTLGLGSAFGQFIWNGSMLAGITALGWLSYASSVAGGGMAMGIGLAGASTGLMEQPLVSRLGVIVGGVILGAGISSTTNSFPIQIKEFLQQSMGTVVQGSIVGRNQDINGIVIPGCFGLSFPCFIGGVAMAGMYGSLMIFPDTKDGKAFPLKYPLAFVMSAFLVIMAMGGTGPQF"""

        EXAMPLE_INSULIN = """>sp|P01308|INS_HUMAN Insulin
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"""

        # Syn3.0: Load full 149-protein FASTA from file, fallback to 5-protein subset
        # On Modal: bundled at /app/bundled/syn30.fasta; locally: ./data/gene_unknown/unknown_aa_seqs.fasta
        SYN30_FASTA_PATHS = ["./bundled/syn30.fasta", "./data/gene_unknown/unknown_aa_seqs.fasta"]
        EXAMPLE_SYN30_FALLBACK = """>MMSYN1_0411 1=Unknown
MQIPIIKPKKAPPLTIEEINEIKQHSSYEKSYLKTFNKYKKKVEHRIYFKTSFWWDIFIIALAALANTITTDYFILATGDTGLFPGGTATIARFLSIVLNKHITSISTSSSFFIFLFIVNLPFFVFGFIKVGIKFTLTSLLYILLSIGWNQIITRLPIINPNEWSLIINYKLISSLPTEWSSKLWLFVFSIFGGFFLGITYSLTYRVGSSTAGTDFISAYVSKKYNKQIGSINMKINFTLLLIFVVLNTVIMPIYKIDSTAKLSVLNTLTDEQFTEIYNKAKDSGKFILDFNSHHHFYLPSNWSVSDQQIWTRQQIAQIIASNTNFTNYDNLTTIIKLKFVFGPSLFASFICFVIQGVVIDRIYPKNKLFTVLISTTKPREVKNYLFESGYRNNIHFLENQTAKKENGYIAQSVIMIHIGLMNWKPLQAGANNIDPDMMISFIRTKQVKGPWSYSLDTQKRELSLYKKVITDRRLMARIEKESILLTKQKITNDKKLKSKSKTF
>MMSYN1_0133 2=Generic
MNNLIVLKGKFEPGKNTKKPNSPQIPKTSIIKLEDCYRILDQLIKASSFWKEQKIDINPIINVKYKRIISKSNRVSYLLLKSLQKNNEHIIGSSFLDELVEKKIVKKQVITYCLTQKDLQEAIKRLDTITNILKKTHFKRIDNNLINLIANEQYLPIKKEIQKYEFLSRTAFISTLVDLNYIEEIFIKTTHIDNNVDSVVTLYDTGIKAIDLLNKLDINVNMSDFIDDYTLFLDRNQYNELKTKAPFLISMSVDDLTKFIIDDKQEEITKNDIISIPDPTNEPIVGVIDTMFCKDVYFSKWVDFRKEVSDDILLDSKDYQHGTQVSSIIVDGPSFNKKLEDGCGRFRVRHFGVMAHSSGNVFSLFKKIKSIVINNLDIKVWNLSLGSIREVSSNYISLLGSLLDQLQYENDVIFIVAGTNDNECKQKIVGSPADSINSIVVNSVDFKNKPANYSRKGPVLTYFNKPDISYYGGVDNNKITVCGCYGEAKVQGTSFAAPWITRKVAYLIYKMNYSKEEAKALIIDSAIKFDKQKDNNRDLIGYGVVPIHINEILQSKNTDIKVLLSYNTKAYYTYNFNLPVPTKENKFPFIAKLTFAYFAESQRSQGVDYTQDELDIQFGPIDNKSESINDINENNQSSSSSNAYIYEYEARKMFAKWNTVKSIIKWSKTNKGKKRQFIKTTNNRWGIRVIRKTRTDNINNKSIKFSLVITFRSIDNKDRIEEFISLCNKSGYWVASKVQIDNKIDIHGKSNEYLDFE
>MMSYN1_0433 1=Unknown
MFLEVIAKDLSDIRVINNSKADRIEFCKNLEVGGLTPSLDEIILANQITLKPLHIMIRNNSKDFFFDDYELIKQLEMISVIQKLPNVHGIVIGALNNDYTINEDFLQRVNKIKGSLKITFNRAFDLVDDPINALNVLVKHKIDTVLTSGGTNLNTGLEVIRQLVDQNLDIQILIGGGVDKNNIKQCLTVNNQIHLGRAARMNSSWNSDISVDEINLFKDLDREQNNE
>MMSYN1_0080 1=Unknown
MAEKQATVYHVTPYDGKWQVKGVGNTRPTKLFDTQKEAIAYANELTKKRQGSVIIHRTTGQVRDSINNKDKKK
>MMSYN1_0005 1=Unknown
MIRDFNNQEVTLDDLEQNNNKTDKNKPKVQFLMRFSLVFSNISTHIFLFVLIVIASLFFGLRYTYYNYKVDLITNAHKIKPSIPKLKEVYKEALQVVEEVKRETDKNSSDSLINKIDEIKTIVKEVTEFANEFNDRSKKVEPKVREVIDQGKKITTDLEKVTKEIEELRKTGDSLTNRVRRGLNNFSTLGNLVGTANNDFKSVNESVIRITDLAKKISEEGKKITANVETIKKEVDYFSKRSEIPLRDIEKLKEIYRQKFPLFERNNKRLQEIWSKLMGIFNQFTVEKTQSNYYNHLIYILLFLIIDSIVLLVLTYMSMISKTMKKILLFYIFGILSFNPFVWVSVVISFLSRPIKNRKRKFS"""

        # Example button handlers
        def load_cas9():
            return EXAMPLE_CAS9
        def load_reca():
            return EXAMPLE_RECA
        def load_cox1():
            return EXAMPLE_COX1
        def load_insulin():
            return EXAMPLE_INSULIN
        def load_syn30():
            for path in SYN30_FASTA_PATHS:
                try:
                    with open(path) as f:
                        return f.read()
                except FileNotFoundError:
                    continue
            return EXAMPLE_SYN30_FALLBACK

        example_btn_cas9.click(fn=load_cas9, outputs=[fasta_text])
        example_btn_reca.click(fn=load_reca, outputs=[fasta_text])
        example_btn_cox1.click(fn=load_cox1, outputs=[fasta_text])
        example_btn_insulin.click(fn=load_insulin, outputs=[fasta_text])
        example_btn_syn30.click(fn=load_syn30, outputs=[fasta_text])

        # Uploading a FASTA file shows its content in the text box above.
        upload_file.upload(fn=load_uploaded_fasta, inputs=[upload_file], outputs=[fasta_text])

        # Export functionality
        def _do_export(fmt, session):
            status, path = export_current_results(fmt, session)
            return _format_export_html(status), path

        def _do_save_emb(session):
            status, path = export_embeddings(session)
            return _format_export_html(status), path

        export_btn.click(
            fn=_do_export,
            inputs=[export_format, session_state],
            outputs=[export_status, export_download],
            show_progress="hidden",
        )

        save_emb_btn.click(
            fn=_do_save_emb,
            inputs=[session_state],
            outputs=[export_status, export_download],
            show_progress="hidden",
        )

        # --- Probability vs. rank plot helper ---
        def _build_prob_plot(session, query_label=None):
            """Build a line plot of exact/partial match probability vs ordered hit rank (1..k)
            using all raw matches from the cache (before threshold filtering)."""
            raw = (session or {}).get("_cache", {}).get("raw_matches", [])
            if not raw:
                return None
            import plotly.graph_objects as go

            # Filter to the selected query
            if query_label and query_label != "All queries":
                qm = [m for m in raw if m.get("query_meta") == query_label]
            else:
                qm = raw

            if not qm:
                return None

            # Sort by D_score descending (rank 1 = best match)
            qm_sorted = sorted(qm, key=lambda m: m.get("D_score", 0), reverse=True)

            ranks = list(range(1, len(qm_sorted) + 1))
            exact_probs = [(m.get("p0", 0) + m.get("p1", 0)) / 2 for m in qm_sorted]
            partial_probs = [(m.get("p0_partial", 0) + m.get("p1_partial", 0)) / 2
                             for m in qm_sorted if m.get("p0_partial") is not None]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ranks,
                y=exact_probs,
                mode="lines",
                name="Exact match",
                line=dict(color="rgba(102, 126, 234, 0.9)", width=2),
                hovertemplate="Ordered hit rank: %{x}<br>Exact probability: %{y:.3f}<extra></extra>",
            ))
            if len(partial_probs) == len(ranks):
                fig.add_trace(go.Scatter(
                    x=ranks,
                    y=partial_probs,
                    mode="lines",
                    name="Partial match",
                    line=dict(color="rgba(118, 75, 162, 0.9)", width=2),
                    hovertemplate="Ordered hit rank: %{x}<br>Partial probability: %{y:.3f}<extra></extra>",
                ))

            title_main = "Exact/Partial Match Probability vs. Ordered Hit Rank"
            if query_label and query_label != "All queries":
                ql = query_label if len(query_label) <= 50 else query_label[:47] + "..."
                title_main = f"Match Probability vs. Ordered Hit Rank — {ql}"
            title = (
                f"{title_main}<br><sup>Hits are sorted by similarity (D_score); rank 1 is the top hit.</sup>"
            )

            fig.update_layout(
                title=title,
                xaxis_title="Ordered Hit Rank (k; 1 = highest similarity)",
                yaxis_title="Calibrated Probability",
                yaxis_range=[0, 1.05],
                height=350,
                margin=dict(l=50, r=20, t=80, b=40),
                template="plotly_white",
                legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
            )
            return fig

        # Mode switching handler
        def update_analysis_mode(mode_choice):
            is_protein_search = mode_choice == "Protein Search (Protein-Vec)"
            return (
                gr.Group(visible=is_protein_search),   # protein_search_params
                gr.Group(visible=not is_protein_search), # clean_params
                gr.Accordion(visible=is_protein_search), # advanced_options
            )

        analysis_mode.change(
            fn=update_analysis_mode,
            inputs=[analysis_mode],
            outputs=[protein_search_params, clean_params, advanced_options],
        )

        # Main prediction submission (dispatches between Protein Search and CLEAN)
        def on_submit(mode, fasta, upload, risk_t, risk_v, max_k, use_pv, custom_emb,
                      lookup, metadata, custom_lookup, custom_meta, m_type,
                      min_prob, hide_unc, c_alpha, fp16_head, session,
                      progress=gr.Progress()):
            _t0 = time.perf_counter()
            if mode == "Enzyme Classification (CLEAN)":
                # CLEAN enzyme classification pipeline
                summary_json, df, session = process_clean_input(
                    fasta, upload, float(c_alpha), session=session,
                    progress=progress,
                    clean_embed_fn=clean_embed_fn,
                )
                # Build per-query dropdown
                if not df.empty and "Query" in df.columns:
                    unique_queries = df["Query"].unique().tolist()
                    choices = ["All queries"] + unique_queries
                else:
                    unique_queries = []
                    choices = ["All queries"]
                count_html = (f'<div style="color:#666;font-size:0.9em;padding:2px 4px;">'
                              f'Showing {len(df)} result{"s" if len(df) != 1 else ""}</div>') if not df.empty else ""
                return (
                    _format_summary_html(summary_json, time.perf_counter() - _t0),
                    df,
                    gr.Dropdown(choices=choices, value=unique_queries[0] if unique_queries else "All queries"),
                    gr.Code(visible=False, value=""),
                    gr.Plot(visible=False),  # No prob plot for CLEAN
                    session,
                    count_html,
                    gr.HTML(visible=False),  # hide empty state
                )
            else:
                # Protein Search pipeline
                # Developer-only local device map; leave commented for public web UI.
                # Public Modal deployments use gi.run_embed_protein_vec = gpu_embed,
                # so no device is passed from the UI.
                # device_map = {
                #     "CPU": "cpu",
                #     "Auto": "auto",
                #     "Apple MPS (experimental)": "mps",
                #     "CUDA": "cuda",
                # }
                # embedding_device = device_map.get(embedding_device_choice, "cpu")
                summary_json, df, session = process_input(
                    "", fasta, upload, "fasta_format", risk_t, risk_v, max_k,
                    use_pv, custom_emb, lookup, metadata, custom_lookup, custom_meta,
                    m_type, min_prob, fp16_head=fp16_head, session=session,
                    progress=progress,
                    embed_fn=embed_fn,
                )
                # Apply hide-uncharacterized filter
                if hide_unc and not df.empty and "Protein Name(s)" in df.columns:
                    df = df[~df["Protein Name(s)"].str.contains("Uncharacterized", case=False, na=False)]
                # Build per-query dropdown choices
                if not df.empty and "Query" in df.columns:
                    unique_queries = df["Query"].unique().tolist()
                    choices = ["All queries"] + unique_queries
                else:
                    unique_queries = []
                    choices = ["All queries"]
                # Build probability vs. rank plot for the first query
                plot_label = unique_queries[0] if unique_queries else None
                fig = _build_prob_plot(session, query_label=plot_label)
                # "Showing N of M" indicator
                total = (session or {}).get("results", {}).get("summary", {}).get("matches_found", 0)
                shown = len(df)
                if shown and total and shown < total:
                    count_html = (f'<div style="color:#666;font-size:0.9em;padding:2px 4px;">'
                                  f'Showing <strong style="color:#444 !important;font-weight:700;">{shown:,}</strong> of '
                                  f'<strong style="color:#444 !important;font-weight:700;">{total:,}</strong> matches in the table below</div>')
                elif shown:
                    count_html = (f'<div style="color:#666;font-size:0.9em;padding:2px 4px;">'
                                  f'<strong style="color:#444 !important;font-weight:700;">{shown:,}</strong> match{"es" if shown != 1 else ""}</div>')
                else:
                    count_html = ""
                return (
                    _format_summary_html(summary_json, time.perf_counter() - _t0),
                    df,
                    gr.Dropdown(choices=choices, value=unique_queries[0] if unique_queries else "All queries"),
                    gr.Code(visible=False, value=""),
                    gr.Plot(value=fig, visible=fig is not None),
                    session,
                    count_html,
                    gr.HTML(visible=False),  # hide empty state
                )

        # Toggle Search <-> Stop around the search so Stop occupies Search's spot
        # only while running (no extra real estate).
        def _searching():
            # Empty card: Gradio's progress text/bar is targeted here via show_progress_on.
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(value='<div style="min-height:48px;"></div>', visible=True),
            )

        def _idle():
            return gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)

        search_event = submit_btn.click(
            fn=_searching,
            inputs=None,
            outputs=[submit_btn, stop_btn, search_status],
            show_progress="hidden",
        ).then(
            fn=on_submit,
            inputs=[
                analysis_mode,
                fasta_text, upload_file,
                risk_type, risk_value, max_results_slider,
                use_protein_vec, custom_embeddings_state,
                lookup_db_state, metadata_db_state, custom_lookup_upload, custom_metadata_upload,
                match_type, min_probability, hide_uncharacterized,
                clean_alpha,
                fp16_head_checkbox,
                session_state,
            ],
            outputs=[results_summary, results_table, query_filter, sequence_detail, prob_plot, session_state, results_count_md, empty_state],
            show_progress="full",
            show_progress_on=[search_status],
        )
        # Revert to Search when the search finishes normally.
        search_event.then(fn=_idle, inputs=None, outputs=[submit_btn, stop_btn, search_status], show_progress="hidden")
        # Stop: revert the button and cancel the in-flight search event.
        stop_btn.click(fn=_idle, inputs=None, outputs=[submit_btn, stop_btn, search_status], cancels=[search_event], show_progress="hidden")

        # Clear button — reset input and results to initial state.
        def _clear():
            return (
                "",  # fasta_text
                None,  # upload_file
                pd.DataFrame(),  # results_table
                gr.Dropdown(choices=["All queries"], value="All queries"),  # query_filter
                gr.HTML(""),  # results_summary
                gr.Code(visible=False, value=""),  # sequence_detail
                gr.Plot(visible=False),  # prob_plot
                {},  # session_state
                gr.HTML(""),  # results_count_md
                gr.HTML(
                    '<div style="text-align:center;padding:40px 20px;color:#999;font-size:1.1em;">'
                    'Run a search to see matched proteins</div>',
                    visible=True,
                ),  # empty_state
                gr.update(value="", visible=False),  # search_status
            )

        clear_btn.click(
            fn=_clear,
            inputs=None,
            outputs=[fasta_text, upload_file, results_table, query_filter,
                     results_summary, sequence_detail, prob_plot,
                     session_state, results_count_md, empty_state, search_status],
            show_progress="hidden",
        )


        # Per-query filtering
        def filter_by_query(query_choice, session):
            if not session or "results" not in session:
                return pd.DataFrame(), gr.Code(visible=False, value=""), gr.Plot(visible=False), ""
            matches = session["results"].get("matches", [])
            if not matches:
                return pd.DataFrame(), gr.Code(visible=False, value=""), gr.Plot(visible=False), ""

            # Filter matches for the selected query
            if query_choice and query_choice != "All queries":
                filtered_matches = [m for m in matches if m.get("query_meta") == query_choice]
            else:
                filtered_matches = matches

            df = pd.DataFrame(filtered_matches)
            # Apply display formatting (same as in _process_input_impl)
            display_header_map = {
                "query_meta": "Query",
                "query_seq": "Query Sequence",
                "lookup_seq": "Match Sequence",
                "lookup_meta": "Match Description",
                "lookup_entry": "UniProt Entry",
                "lookup_pfam": "Pfam",
                "lookup_protein_names": "Protein Name(s)",
                "prob_exact": "Exact Prob",
                "prob_partial": "Partial Prob",
            }
            preferred_order = [
                "query_meta", "lookup_entry", "lookup_protein_names",
                "lookup_pfam", "prob_exact", "prob_partial",
            ]
            HIDDEN_COLS = {"D_score", "p0", "p1", "p0_partial", "p1_partial", "query_seq", "lookup_seq", "lookup_meta"}
            display_columns = [col for col in preferred_order if col in df.columns
                               and col not in HIDDEN_COLS]
            display_columns.extend([col for col in df.columns if col not in display_columns
                                    and col not in HIDDEN_COLS])
            display_df = df.reindex(columns=display_columns).copy()
            display_df = display_df.rename(columns=display_header_map)
            _truncate_display_columns(display_df)

            # Build probability plot for the filtered query
            plot_label = query_choice if query_choice and query_choice != "All queries" else None
            fig = _build_prob_plot(session, query_label=plot_label)
            # "Showing N of M" for the filtered view
            total = len(matches)
            shown = len(display_df)
            if shown and total and shown < total:
                count_md = (f'<div style="color:#666;font-size:0.9em;padding:2px 4px;">'
                            f'Showing <strong style="color:#444 !important;font-weight:700;">{shown:,}</strong> of '
                            f'<strong style="color:#444 !important;font-weight:700;">{total:,}</strong> matches in the table below</div>')
            elif shown:
                count_md = (f'<div style="color:#666;font-size:0.9em;padding:2px 4px;">'
                            f'<strong style="color:#444 !important;font-weight:700;">{shown:,}</strong> match{"es" if shown != 1 else ""}</div>')
            else:
                count_md = ""
            return (
                display_df,
                gr.Code(visible=False, value=""),
                gr.Plot(value=fig, visible=fig is not None),
                count_md,
            )

        query_filter.change(
            fn=filter_by_query,
            inputs=[query_filter, session_state],
            outputs=[results_table, sequence_detail, prob_plot, results_count_md],
            show_progress="hidden",
        )

        display_to_match_key = {
            "Query": "query_meta",
            "Match Sequence": "lookup_seq",
            "UniProt Entry": "lookup_entry",
            "Protein Name(s)": "lookup_protein_names",
            "Pfam": "lookup_pfam",
            "Match Description": "lookup_meta",
            "Exact Prob": "prob_exact",
            "Partial Prob": "prob_partial",
            "Exact Match Prob": "prob_exact",
            "Partial Match Prob": "prob_partial",
        }

        def normalize_value(value):
            if value is None:
                return ""
            try:
                if pd.isna(value):
                    return ""
            except Exception:
                pass
            return str(value).strip()

        def row_dict_from_event(df, evt: gr.SelectData, row_idx: int) -> Dict[str, Any]:
            row_value = getattr(evt, "row_value", None)
            if isinstance(row_value, dict):
                return row_value
            if isinstance(row_value, (list, tuple)):
                return {
                    col: row_value[i] if i < len(row_value) else None
                    for i, col in enumerate(df.columns)
                }
            if 0 <= row_idx < len(df):
                return df.iloc[row_idx].to_dict()
            return {}

        def parse_selection_indices(evt: gr.SelectData) -> Tuple[int, int]:
            row_idx, col_idx = -1, -1
            index = getattr(evt, "index", None)
            if isinstance(index, (list, tuple)):
                if len(index) >= 1:
                    row_idx = index[0]
                if len(index) >= 2:
                    col_idx = index[1]
            else:
                row_idx = index

            try:
                row_idx = int(row_idx)
            except (TypeError, ValueError):
                row_idx = -1
            try:
                col_idx = int(col_idx)
            except (TypeError, ValueError):
                col_idx = -1

            return row_idx, col_idx

        def find_full_match(display_row: Dict[str, Any], matches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not display_row or not matches:
                return None

            selected = {}
            for display_col, match_key in display_to_match_key.items():
                if display_col not in display_row:
                    continue
                normalized = normalize_value(display_row.get(display_col))
                if normalized:
                    selected[match_key] = normalized

            if not selected:
                return None

            match_keys = [
                "query_meta",
                "lookup_entry",
                "lookup_seq",
                "lookup_protein_names",
                "lookup_pfam",
                "lookup_meta",
            ]
            candidates = matches
            for key in match_keys:
                selected_value = selected.get(key)
                if not selected_value:
                    continue
                narrowed = [m for m in candidates if normalize_value(m.get(key)) == selected_value]
                if narrowed:
                    candidates = narrowed

            return candidates[0] if candidates else None

        # Row selection → show full sequences from session
        def on_row_select(df, session, evt: gr.SelectData):
            if df is None or df.empty:
                return gr.Code(visible=False, value="")

            row_idx, _ = parse_selection_indices(evt)

            display_row = row_dict_from_event(df, evt, row_idx)

            matches = (session or {}).get("results", {}).get("matches", [])

            m = find_full_match(display_row, matches)
            if m is None and 0 <= row_idx < len(matches):
                # Best-effort fallback for older Gradio event payloads
                m = matches[row_idx]

            if m:
                parts = []
                if m.get("lookup_protein_names"):
                    parts.append(f"Protein: {m['lookup_protein_names']}")
                if m.get("lookup_entry"):
                    parts.append(f"Entry: {m['lookup_entry']}")
                if m.get("lookup_pfam"):
                    parts.append(f"Pfam: {m['lookup_pfam']}")
                exact_prob = m.get("prob_exact")
                if not exact_prob and m.get("p0") is not None and m.get("p1") is not None:
                    mean_e = (float(m["p0"]) + float(m["p1"])) / 2.0
                    half_e = abs(float(m["p1"]) - float(m["p0"])) / 2.0
                    exact_prob = f"{mean_e:.3f} ± {half_e:.3f}"
                partial_prob = m.get("prob_partial")
                if (
                    not partial_prob
                    and m.get("p0_partial") is not None
                    and m.get("p1_partial") is not None
                ):
                    mean_p = (float(m["p0_partial"]) + float(m["p1_partial"])) / 2.0
                    half_p = abs(float(m["p1_partial"]) - float(m["p0_partial"])) / 2.0
                    partial_prob = f"{mean_p:.3f} ± {half_p:.3f}"
                if exact_prob:
                    parts.append(f"Exact Prob: {exact_prob}")
                if partial_prob:
                    parts.append(f"Partial Prob: {partial_prob}")
                if m.get("query_meta"):
                    parts.append(f"\nQuery: {m['query_meta']}")
                if m.get("query_seq"):
                    parts.append(f"\nQuery Sequence:\n{m['query_seq']}")
                if m.get("lookup_seq"):
                    parts.append(f"\nMatch Sequence:\n{m['lookup_seq']}")
                text = "\n".join(parts)
            else:
                # Fallback to display table row
                if not display_row and 0 <= row_idx < len(df):
                    display_row = df.iloc[row_idx].to_dict()
                parts = [f"{col}: {value}" for col, value in display_row.items() if normalize_value(value)]
                text = "\n\n".join(parts) if parts else "No detail available"

            return gr.Code(visible=True, value=text)

        results_table.select(
            fn=on_row_select,
            inputs=[results_table, session_state],
            outputs=[sequence_detail],
        )

        # Database selection event handler
        def update_database_selection(db_choice):
            if db_choice == "Swiss-Prot (540K)":
                return (
                    DEFAULT_LOOKUP_EMBEDDING,
                    DEFAULT_LOOKUP_METADATA,
                    gr.File(value=None, visible=False),
                    gr.File(value=None, visible=False),
                )
            if db_choice == "SCOPE":
                return (
                    DEFAULT_SCOPE_EMBEDDING,
                    DEFAULT_SCOPE_METADATA,
                    gr.File(value=None, visible=False),
                    gr.File(value=None, visible=False),
                )
            if db_choice == "AFDB (Clustered)":
                return (
                    DEFAULT_AFDB_EMBEDDING,
                    DEFAULT_AFDB_METADATA,
                    gr.File(value=None, visible=False),
                    gr.File(value=None, visible=False),
                )
            if db_choice == "Euk (74K)":
                return (
                    DEFAULT_EUK_EMBEDDING,
                    DEFAULT_EUK_METADATA,
                    gr.File(value=None, visible=False),
                    gr.File(value=None, visible=False),
                )
            # Custom database
            return (
                CUSTOM_UPLOAD_EMBEDDING,
                CUSTOM_UPLOAD_METADATA,
                gr.File(visible=True),
                gr.File(visible=True),
            )

        db_type.change(
            fn=update_database_selection,
            inputs=[db_type],
            outputs=[lookup_db_state, metadata_db_state, custom_lookup_upload, custom_metadata_upload]
        )

        # Update alpha dropdown / probability slider when risk type changes
        def update_risk_controls(risk_choice):
            if risk_choice == "Probability Filter":
                return (
                    gr.Dropdown(choices=FDR_ALPHAS, value="0.1", visible=False),
                    gr.Slider(visible=True),
                )
            elif risk_choice == "FNR":
                return (
                    gr.Dropdown(choices=FNR_ALPHAS, value="0.1", visible=True),
                    gr.Slider(visible=False),
                )
            else:  # FDR
                return (
                    gr.Dropdown(choices=FDR_ALPHAS, value="0.1", visible=True),
                    gr.Slider(visible=False),
                )

        risk_type.change(
            fn=update_risk_controls,
            inputs=[risk_type],
            outputs=[risk_value, min_probability]
        )
    
    # Gradio 6 expects theme/css to be passed at launch(), not Blocks().
    # Attach them so the launcher can pass them without create_interface()
    # returning a custom tuple.
    interface.cpr_theme = brand_theme
    interface.cpr_css = custom_css
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=False,
        theme=getattr(interface, "cpr_theme", None),
        css=getattr(interface, "cpr_css", None),
    )
