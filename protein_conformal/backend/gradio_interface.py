"""
Backend Gradio interface for Protein Conformal Prediction.

This module provides the Gradio web interface for the protein conformal prediction framework.
It allows users to input protein sequences or FASTA files, generate embeddings,
and perform conformal prediction with statistical guarantees.
"""

import gradio as gr
import numpy as np
import os
import sys
import tempfile
import io
import logging
import shutil
import time
import threading
from functools import lru_cache
import pandas as pd
from Bio import SeqIO
from typing import List, Union, Tuple, Dict, Optional, Any, Set

from protein_conformal.util import load_database, query, read_fasta, get_sims_labels, get_thresh_new_FDR, get_thresh_new, risk, calculate_false_negatives, simplifed_venn_abers_prediction


logger = logging.getLogger(__name__)


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

# Paths used for temporary storage of uploaded database files
CUSTOM_UPLOAD_EMBEDDING = "./data/custom_uploaded_embedding.npy" # path to the custom uploaded embeddings
CUSTOM_UPLOAD_METADATA = "./data/custom_uploaded_metadata.tsv" # path to the custom uploaded metadata

# Amino acid validation constants
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')
SPECIAL_CHARS = set('XUB')  # X=unknown, U=selenocysteine, B=ambiguous D/N

# Global session storage for the current Gradio instance
CURRENT_SESSION = {}


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
# repeating the expensive np.load + load_database work.
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

    embeddings = np.load(embedding_path, allow_pickle=True).astype(np.float32, copy=False)
    metadata_kind, lookup_seqs, lookup_meta = _load_lookup_metadata(metadata_path)
    lookup_index = load_database(embeddings.copy())

    resource = {
        "index": lookup_index,
        "lookup_seqs": lookup_seqs,
        "lookup_meta": lookup_meta,
        "metadata_kind": metadata_kind,
        "num_embeddings": embeddings.shape[0],
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

def process_uploaded_file(file_obj) -> Tuple[List[str], List[str]]:
    """
    Process an uploaded FASTA file from Gradio's File component.

    Supports the different object types that Gradio may hand over:
    - file-like objects exposing .read()
    - temporary files with a .name attribute
    - plain filesystem paths (default when type='filepath')
    - dictionaries containing 'path'/'name' metadata
    """
    if file_obj is None:
        return [], []

    def _read_text(handle) -> str:
        data = handle.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        return data

    fasta_text: Optional[str] = None

    if hasattr(file_obj, "read"):
        fasta_text = _read_text(file_obj)
    else:
        candidate_paths: List[Optional[str]] = []
        if isinstance(file_obj, dict):
            candidate_paths.extend([file_obj.get("path"), file_obj.get("name")])
        else:
            candidate_paths.append(getattr(file_obj, "name", None))
            if isinstance(file_obj, str):
                candidate_paths.append(file_obj)

        for path in candidate_paths:
            if not path:
                continue
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    fasta_text = fh.read().decode("utf-8", errors="replace")
                break

    if not fasta_text:
        raise AttributeError("Uploaded FASTA file is not readable and has no accessible path.")

    return parse_fasta(fasta_text)

def run_embed_protein_vec(sequences: List[str], progress=gr.Progress()) -> np.ndarray:
    """
    use existing embed_protein_vec.py to generate embeddings
    
    Args:
        sequences: List of protein sequences
        progress: Gradio progress bar
        
    Returns:
        NumPy array of embeddings
    """
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
        progress(0.2, desc="Running embed_protein_vec.py...")
        
        # Run the embed_protein_vec.py script
        cmd = [
            sys.executable, 
            "protein_conformal/embed_protein_vec.py",
            "--input_file", tmp_fasta_path,
            "--output_file", tmp_out_path,
            "--path_to_protein_vec", "protein_vec_models"
        ]
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            raise Exception(f"embed_protein_vec.py failed: {result.stderr}")
        
        progress(0.8, desc="Loading embeddings...")
        # Load the embeddings
        embeddings = np.load(tmp_out_path)
        
        progress(1.0, desc="Embeddings complete!")
        return embeddings
        
    finally:
        # Clean up temporary files
        if os.path.exists(tmp_fasta_path):
            os.unlink(tmp_fasta_path)
        if os.path.exists(tmp_out_path):
            os.unlink(tmp_out_path)

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
               progress=gr.Progress()) -> pd.DataFrame:
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
    progress(0.1, desc="Preparing lookup database...")
    resources = get_lookup_resources(lookup_embedding_path, lookup_metadata_path)
    lookup_database = resources["index"]
    lookup_seqs = resources["lookup_seqs"]
    lookup_meta = resources["lookup_meta"]
    metadata_kind = resources["metadata_kind"]
    max_neighbors = min(k, resources["num_embeddings"])
    
    progress(0.7, desc="Running search...")
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
    
    progress(1.0, desc="Search completed!")
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
                  progress=gr.Progress()) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Wrapper that instruments the main pipeline with timing information."""
    stage_timer = StageTimer()
    try:
        return _process_input_impl(
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
            progress,
        )
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
                        progress=gr.Progress()) -> Tuple[Dict[str, Any], pd.DataFrame]:
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
    # Step 1: Get sequences and metadata from FASTA input
    with stage_timer.track("parse_input"):
        sequences = []
        query_meta = []
        if upload_file is not None:
            sequences, query_meta = process_uploaded_file(upload_file)
        elif fasta_text and fasta_text.strip():
            sequences, query_meta = parse_fasta(fasta_text)
        else:
            return {"error": "No FASTA input provided. Please enter FASTA content or upload a FASTA file."}, pd.DataFrame()

    if not sequences and custom_embeddings is None:
        return {"error": "No sequences found in the FASTA input. Please check your input format."}, pd.DataFrame()

    # Ensure conformal_results is initialized
    conformal_results = {}

    # Step 2: Get embeddings
    if use_protein_vec and not custom_embeddings:
        try:
            progress(0.1, desc="Starting embedding process...")
            with stage_timer.track("protein_vec_embedding"):
                embeddings = run_embed_protein_vec(sequences, progress)
            progress(0.6, desc="Embeddings complete!")
        except Exception as e:
            return {"error": f"Error generating embeddings: {str(e)}"}, pd.DataFrame()
    elif custom_embeddings:
        try:
            progress(0.2, desc="Loading custom embeddings...")
            with stage_timer.track("load_custom_embeddings"):
                # Load user-provided embeddings
                with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
                    tmp.write(custom_embeddings.read())
                    tmp_path = tmp.name
                
                embeddings = np.load(tmp_path)
                os.unlink(tmp_path)  # Clean up temp file
            progress(0.4, desc="Custom embeddings loaded!")
        except Exception as e:
            return {"error": f"Error loading embeddings: {str(e)}"}, pd.DataFrame()
    else:
        return {"error": "Either Protein-Vec must be enabled or custom embeddings must be provided"}, pd.DataFrame()
    
    # Handle custom uploaded database files if present
    if custom_lookup_upload is not None:
        try:
            lookup_db = _persist_uploaded_file(custom_lookup_upload, CUSTOM_UPLOAD_EMBEDDING)
        except Exception as e:
            return {"error": f"Error processing custom database: {str(e)}"}, pd.DataFrame()

    if custom_metadata_upload is not None:
        try:
            metadata_db = _persist_uploaded_file(custom_metadata_upload, CUSTOM_UPLOAD_METADATA, preserve_suffix=True)
        except Exception as e:
            return {"error": f"Error processing custom metadata: {str(e)}"}, pd.DataFrame()
    
    # Step 3: Perform conformal prediction
    try:
        # Determine which database is being used
        database_type = "Custom"
        if lookup_db == DEFAULT_LOOKUP_EMBEDDING:
            database_type = "UniProt"
        elif lookup_db == DEFAULT_SCOPE_EMBEDDING:
            database_type = "SCOPE"
        
        progress(0.5, desc=f"Performing conformal prediction with {risk_type} control...")
        
        with stage_timer.track("threshold_lookup"):
            # Determine match type suffix for file/column lookup
            is_partial = match_type.lower() == "partial"
            match_type_label = "partial" if is_partial else "exact"

            if risk_type.lower() == "fdr":
                # Try partial FDR file if requested, fall back to exact
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
                # Look for match-type-specific FDR column
                fdr_col = 'partial_fdr' if is_partial else 'exact_fdr'
                empirical_risk = threshold_df.iloc[closest_idx].get(fdr_col, None)
                risk_description = "False Discovery Rate"
                risk_formula = "FDR = FP / (FP + TP)"
                risk_explanation = ("Controls the proportion of false discoveries (incorrect matches) "
                                      "among all retrieved matches. Useful when you want to ensure most "
                                      "retrieved results are correct.")
                if abs(closest_alpha - risk_value) > 0.001:
                    logger.warning(f"Requested alpha={risk_value} not available. Using closest alpha={closest_alpha}")
            else:
                # Try partial FNR file if requested, fall back to exact
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
                # Look for match-type-specific FNR column
                fnr_col = 'partial_fnr' if is_partial else 'exact_fnr'
                empirical_risk = threshold_df.iloc[closest_idx].get(fnr_col, None)
                risk_description = "False Negative Rate"
                risk_formula = "FNR = FN / (FN + TP)"
                risk_explanation = ("Controls the proportion of missed true matches among all actual matches. "
                                      "Useful when you want to minimize missing true relationships, even at the "
                                      "cost of including some false positives.")
                if abs(closest_alpha - risk_value) > 0.001:
                    logger.warning(f"Requested alpha={risk_value} not available. Using closest alpha={closest_alpha}")
            
        conformal_results = {
            "message": f"Used precomputed threshold for {risk_description} control, alpha={closest_alpha}, match_type={match_type_label}" + (f" (requested α={risk_value})" if abs(closest_alpha - risk_value) > 0.001 else ""),
            "threshold": float(threshold),
            "risk_type": risk_type.lower(),
            "match_type": match_type_label,
            "risk_description": risk_description,
            "risk_formula": risk_formula,
            "risk_explanation": risk_explanation,
            "empirical_risk": float(empirical_risk) if empirical_risk is not None else None,
            "n_matches": 0,  # Will be updated after search
            "match_rate": 0.0,  # Will be updated after search
            "n_calib": len(threshold_df),
            "match_info": [],  # Will be populated after search
            "has_probability_calibration": True,  # Enable Venn-Abers probability calibration
            "probability_calibration_method": "Venn-Abers prediction with isotonic regression"
        }

        if "error" in conformal_results:
            return conformal_results, pd.DataFrame()

        # Step 4: Run the search against the database with the threshold
        progress(0.7, desc=f"Searching database with conformal threshold...")
        
        with stage_timer.track("database_search"):
            results_df = run_search(
                embeddings,
                sequences,
                query_meta,
                lookup_db,
                metadata_db,
                threshold=conformal_results.get("threshold", 0.0),
                k=max_results,
                progress=progress
            )
        
        # Process the results
        all_matches = results_df.to_dict(orient="records")
        
        # If probability calibration is available, add probabilities to the matches
        if conformal_results["has_probability_calibration"]:
            # Using precomputed calibration data to interpolate probabilities
            progress(0.8, desc="Calibrating probabilities...")
            with stage_timer.track("probability_calibration"):
                try:
                    # Load precomputed calibration data
                    cal_df = load_results_dataframe(
                        DEFAULT_CALIBRATION_DATA,
                        required_columns=["similarity", "prob_exact_p0", "prob_exact_p1"],
                    )

                    # Sort by similarity for interpolation
                    cal_df = cal_df.sort_values('similarity')
                    sim_cal = cal_df['similarity'].values
                    p0_cal = cal_df['prob_exact_p0'].values
                    p1_cal = cal_df['prob_exact_p1'].values

                    # Calculate probabilities for each match using interpolation
                    for i, match in enumerate(all_matches):
                        sim_score = match["D_score"]
                        # Interpolate p0 and p1 from calibration data
                        p0 = np.interp(sim_score, sim_cal, p0_cal)
                        p1 = np.interp(sim_score, sim_cal, p1_cal)
                        # Average the probabilities as in the paper
                        all_matches[i]["prob_exact"] = (p0 + p1) / 2
                        all_matches[i]["p0"] = float(p0)
                        all_matches[i]["p1"] = float(p1)
                except Exception as e:
                    logger.warning(f"Probability calibration failed: {e}. Using similarity scores as proxy.")
                    for i, match in enumerate(all_matches):
                        all_matches[i]["prob_exact"] = match["D_score"]
        else:
            # If no probability calibration, use similarity as a proxy
            for i, match in enumerate(all_matches):
                all_matches[i]["prob_exact"] = match["D_score"]
        
        # Store all matches in the session
        total_matches = len(all_matches)
        
        # Convert all_matches back to DataFrame with prob_exact column
        results_df = pd.DataFrame(all_matches)
        
        with stage_timer.track("results_packaging"):
            # 1. Create summary information for the JSON display
            summary = {
                "message": conformal_results["message"],
                "database_used": database_type,
                "num_matches": total_matches,
                "threshold": conformal_results["threshold"],
                "max_results_requested": max_results,
                "risk_type": conformal_results["risk_type"],
                "risk_description": conformal_results["risk_description"],
                "risk_formula": conformal_results["risk_formula"], 
                "risk_explanation": conformal_results["risk_explanation"],
                "empirical_risk": conformal_results["empirical_risk"],
                "n_calib": conformal_results["n_calib"],
                "match_rate": conformal_results["match_rate"],
            }
            
            # 2. Complete results for the raw JSON output
            complete_results = {
                "conformal_prediction": conformal_results,
                "database_used": database_type,
                "num_matches": total_matches,
                "max_results_requested": max_results,
                "matches": all_matches,  # Include all matches
                "threshold": conformal_results["threshold"],
                "risk_type": conformal_results["risk_type"],
                "risk_description": conformal_results["risk_description"],
                "risk_explanation": conformal_results["risk_explanation"],
                "empirical_risk": conformal_results["empirical_risk"]
            }
            
            # Store in session for potential later use - include ALL matches for export
            global CURRENT_SESSION
            CURRENT_SESSION = {
                "results": complete_results,
                "parameters": {
                    "risk_type": risk_type,
                    "risk_value": risk_value,
                    "max_results": max_results,
                    "database_type": database_type,
                    "input_type": "fasta_format",  # Always FASTA now
                    "n_calib": conformal_results["n_calib"]
                }
            }
        
        progress(1.0, desc="Conformal prediction complete!")
        display_header_map = {
            "query_seq": "Query Sequence",
            "query_meta": "Query Description",
            "lookup_seq": "Match Sequence",
            "lookup_meta": "Match Description",
            "lookup_entry": "UniProt Entry",
            "lookup_pfam": "Pfam",
            "lookup_protein_names": "Protein Name(s)",
            "D_score": "Similarity (D score)",
            "prob_exact": "Match Probability",
            "p0": "Prob (p0)",
            "p1": "Prob (p1)",
        }
        preferred_order = [
            "query_meta",
            "query_seq",
            "lookup_entry",
            "lookup_pfam",
            "lookup_protein_names",
            "lookup_meta",
            "lookup_seq",
            "D_score",
            "prob_exact",
            "p0",
            "p1",
        ]
        display_columns = [col for col in preferred_order if col in results_df.columns]
        display_columns.extend([col for col in results_df.columns if col not in display_columns])
        display_df = results_df.reindex(columns=display_columns).rename(columns=display_header_map)
        return summary, display_df
    except Exception as e:
        error_message = {"error": f"Error during search: {str(e)}"}
        return error_message, pd.DataFrame()

def export_current_results(format_type: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Export the current results in the specified format.
    All matches (not just displayed ones) will be included in the export.
    
    Args:
        format_type: Format to export (csv, json)
        
    Returns:
        Tuple of (status dict, file path for download)
    """
    global CURRENT_SESSION
    
    if not CURRENT_SESSION or "results" not in CURRENT_SESSION:
        return {"error": "No results to export"}, None
    
    try:
        # Create a directory for exported reports if it doesn't exist
        os.makedirs("exported_reports", exist_ok=True)
        
        # Create a unique filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        risk_type = (CURRENT_SESSION.get("parameters", {}).get("risk_type") or "risk").lower()
        threshold = CURRENT_SESSION.get("results", {}).get("threshold")
        threshold_tag = "thr_unknown"
        if isinstance(threshold, (int, float)):
            threshold_tag = f"thr_{threshold:.4f}".replace(".", "p")
        file_path = os.path.join(
            "exported_reports",
            f"results_{timestamp}_{risk_type}_{threshold_tag}.{format_type}",
        )
        
        # Export the results
        if format_type == "csv":
            if "matches" in CURRENT_SESSION["results"]:
                import pandas as pd
                # Export ALL matches, not just the displayed ones
                df = pd.DataFrame(CURRENT_SESSION["results"]["matches"])
                df.to_csv(file_path, index=False)
                total_exported = len(df)
            else:
                return {"error": "No matches to export"}, None
        elif format_type == "json":
            with open(file_path, 'w') as f:
                import json
                # For JSON export, we include the full result structure
                json.dump(CURRENT_SESSION["results"], f, indent=2)
                total_exported = CURRENT_SESSION["results"]["num_matches"]
        else:
            return {"error": f"Unsupported format: {format_type}"}, None
        
        return {
            "success": True,
            "message": f"Results exported as {file_path} ({total_exported} records)",
            "file_path": file_path
        }, file_path
    
    except Exception as e:
        return {
            "error": f"Error exporting results: {str(e)}"
        }, None

def create_interface():
    """
    Create and configure the Gradio interface for protein conformal prediction

    Returns:
        Gradio interface object
    """
    # Custom CSS for better styling
    custom_css = """
    .header-container {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .header-title {
        color: white;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1em;
    }
    .info-box {
        background-color: #f0f7ff;
        border-left: 4px solid #667eea;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #fff7e6;
        border-left: 4px solid #faad14;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .success-box {
        background-color: #f6ffed;
        border-left: 4px solid #52c41a;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .parameter-card {
        background: #fafafa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e8e8e8;
    }
    """

    with gr.Blocks(title="Conformal Protein Retrieval", css=custom_css, theme=gr.themes.Soft()) as interface:
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; font-size: 2.2em; margin-bottom: 5px;">Conformal Protein Retrieval</h1>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.1em; margin: 0;">
                Functional protein mining with statistical guarantees
            </p>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9em; margin-top: 5px;">
                Based on <a href="https://www.nature.com/articles/s41467-024-55676-y" target="_blank" style="color: #fff;">Boger et al., Nature Communications 2025</a>
            </p>
        </div>
        """)

        # Quick info box
        gr.HTML("""
        <div style="background-color: #f0f7ff; border-left: 4px solid #667eea; padding: 12px 15px; margin: 10px 0; border-radius: 0 8px 8px 0;">
            <strong>How it works:</strong> Enter protein sequences, select your risk control type (FDR or FNR),
            and get statistically guaranteed results with calibrated match probabilities.
        </div>
        """)

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
                            placeholder=">sp|P00533|EGFR_HUMAN Epidermal growth factor receptor\nMRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLE\n\n>sp|P04637|P53_HUMAN Cellular tumor antigen p53\nMEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"
                        )

                        upload_file = gr.File(
                            label="Or Upload FASTA File",
                            file_types=[".fasta", ".fa", ".txt"]
                        )

                        # Example buttons
                        gr.Markdown("**Quick Examples:**")
                        with gr.Row():
                            example_btn1 = gr.Button("EGFR", size="sm")
                            example_btn2 = gr.Button("P53", size="sm")
                            example_btn3 = gr.Button("Insulin", size="sm")

                        use_protein_vec = gr.State(value=True)
                        custom_embeddings_state = gr.State(value=None)

                    # Right column - Parameters
                    with gr.Column(scale=1):
                        gr.Markdown("### Search Parameters")

                        with gr.Group():
                            risk_type = gr.Radio(
                                ["FDR", "FNR"],
                                label="Risk Control Type",
                                value="FDR",
                                info="FDR: Minimize false positives | FNR: Minimize missed matches"
                            )

                            risk_value = gr.Slider(
                                minimum=0.001,
                                maximum=0.2,
                                value=0.1,
                                step=0.001,
                                label="Risk Level (α)",
                                info="Lower = stricter threshold, fewer but more confident results"
                            )

                            match_type = gr.Radio(
                                ["Exact", "Partial"],
                                label="Match Type",
                                value="Exact",
                                info="Exact: All Pfam domains match | Partial: At least one matches"
                            )

                        # Calibrated thresholds info
                        gr.HTML("""
                        <div style="background-color: #fff7e6; border-left: 4px solid #faad14; padding: 10px 12px; margin: 10px 0; border-radius: 0 8px 8px 0; font-size: 0.9em;">
                            <strong>Available thresholds:</strong><br>
                            • FDR: α=0.1 only (paper-verified)<br>
                            • FNR: α ∈ {0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2}
                        </div>
                        """)

                        # Database options in accordion
                        with gr.Accordion("Advanced Options", open=False):
                            db_type = gr.Radio(
                                ["UniProt", "SCOPE", "Custom"],
                                label="Database",
                                value="UniProt",
                                info="Select lookup database"
                            )

                            max_results_slider = gr.Slider(
                                minimum=100,
                                maximum=5000,
                                value=1000,
                                step=100,
                                label="Max Results (k)",
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

                        lookup_db_state = gr.State(value=DEFAULT_LOOKUP_EMBEDDING)
                        metadata_db_state = gr.State(value=DEFAULT_LOOKUP_METADATA)

                        # Submit button
                        submit_btn = gr.Button("Search", variant="primary", size="lg")

                # Results section
                gr.Markdown("### Results")

                with gr.Row():
                    with gr.Column(scale=2):
                        results_table = gr.Dataframe(
                            label="Matches (click headers to sort)",
                            wrap=True,
                            interactive=True,
                            height=400
                        )

                    with gr.Column(scale=1):
                        results_summary = gr.JSON(label="Search Summary")

                        with gr.Row():
                            export_format = gr.Radio(
                                ["csv", "json"],
                                label="Export Format",
                                value="csv",
                                scale=2
                            )
                            export_btn = gr.Button("Export", size="sm", scale=1)

                        export_status = gr.JSON(label="Export Status", visible=True)
                        export_download = gr.File(label="Download", interactive=False)

            with gr.TabItem("About"):
                gr.Markdown("""
                ## What is Conformal Prediction?

                Conformal prediction provides **mathematically rigorous uncertainty quantification** for machine learning predictions.

                ### Risk Control Types

                | Type | Formula | Use When |
                |------|---------|----------|
                | **FDR** | FP / (FP + TP) | You want high precision - most retrieved results should be correct |
                | **FNR** | FN / (FN + TP) | You want high recall - don't miss true matches |

                ### How It Works

                1. **Embedding**: Your protein sequences are converted to 512-dimensional vectors using Protein-Vec
                2. **Search**: FAISS finds the k-nearest neighbors in the lookup database
                3. **Thresholding**: Conformal prediction selects a similarity threshold λ̂ that guarantees:
                   - E[Risk] ≤ α (your specified risk level)
                4. **Calibration**: Venn-Abers provides calibrated probability estimates for each match

                ### The Math

                The threshold is chosen as:
                ```
                λ̂ = inf{λ: (1/n)∑ℓ(Xᵢ, Cλ(Xᵢ)) ≤ α - (1-α)/n}
                ```

                This guarantees that the expected risk E[ℓ(X, Cλ̂(X))] ≤ α under exchangeability.

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
        EXAMPLE_EGFR = """>sp|P00533|EGFR_HUMAN Epidermal growth factor receptor
MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLE"""

        EXAMPLE_P53 = """>sp|P04637|P53_HUMAN Cellular tumor antigen p53
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQ"""

        EXAMPLE_INSULIN = """>sp|P01308|INS_HUMAN Insulin
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"""

        # Example button handlers
        def load_egfr():
            return EXAMPLE_EGFR
        def load_p53():
            return EXAMPLE_P53
        def load_insulin():
            return EXAMPLE_INSULIN

        example_btn1.click(fn=load_egfr, outputs=[fasta_text])
        example_btn2.click(fn=load_p53, outputs=[fasta_text])
        example_btn3.click(fn=load_insulin, outputs=[fasta_text])

        # Export functionality
        export_btn.click(
            fn=export_current_results,
            inputs=[export_format],
            outputs=[export_status, export_download]
        )

        # Main prediction submission
        submit_btn.click(
            fn=lambda fasta, upload, risk_t, risk_v, max_k, use_pv, custom_emb, lookup, metadata, custom_lookup, custom_meta, m_type:
                process_input("", fasta, upload, "fasta_format", risk_t, risk_v, max_k, use_pv, custom_emb, lookup, metadata, custom_lookup, custom_meta, m_type),
            inputs=[
                fasta_text, upload_file,
                risk_type, risk_value, max_results_slider,
                use_protein_vec, custom_embeddings_state,
                lookup_db_state, metadata_db_state, custom_lookup_upload, custom_metadata_upload,
                match_type
            ],
            outputs=[results_summary, results_table]
        )

        # Database selection event handler
        def update_database_selection(db_choice):
            if db_choice == "UniProt":
                return (
                    DEFAULT_LOOKUP_EMBEDDING,
                    DEFAULT_LOOKUP_METADATA,
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                )
            if db_choice == "SCOPE":
                return (
                    DEFAULT_SCOPE_EMBEDDING,
                    DEFAULT_SCOPE_METADATA,
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                )
            # Custom database
            return (
                CUSTOM_UPLOAD_EMBEDDING,
                CUSTOM_UPLOAD_METADATA,
                gr.update(visible=True),
                gr.update(visible=True),
            )

        db_type.change(
            fn=update_database_selection,
            inputs=[db_type],
            outputs=[lookup_db_state, metadata_db_state, custom_lookup_upload, custom_metadata_upload]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=False)
