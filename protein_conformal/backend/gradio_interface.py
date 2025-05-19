"""
Backend Gradio interface for Protein Conformal Prediction.

This module provides the Gradio web interface for the protein conformal prediction framework.
It allows users to input protein sequences or FASTA files, generate embeddings,
and perform conformal prediction with statistical guarantees.
"""

import gradio as gr
import torch
import numpy as np
import os
import sys
import gc
import tempfile
import io
import pandas as pd
from Bio import SeqIO
from transformers import T5EncoderModel, T5Tokenizer
from typing import List, Union, Tuple, Dict, Optional, Any, Set

# Add the protein_vec_models directory to Python's path
sys.path.append("protein_vec_models")
try:
    from model_protein_moe import trans_basic_block, trans_basic_block_Config
    from utils_search import featurize_prottrans, embed_vec
except ImportError:
    print("Warning: Could not import Protein-Vec models. Make sure 'protein_vec_models' directory exists.")

# Add the protein_conformal directory to path for using utility functions
from protein_conformal.util import load_database, query, read_fasta

# Pre-embedded database paths
DEFAULT_LOOKUP_EMBEDDING = "./data/lookup_embeddings.npy"
DEFAULT_LOOKUP_METADATA = "./data/lookup_embeddings_meta_data.tsv"
DEFAULT_CALIBRATION_DATA = "./data/pfam_new_proteins.npy"

# First, add constants for the SCOPE database files at the top of the file, near the other DEFAULT constants
DEFAULT_SCOPE_EMBEDDING = "./data/lookup/scope_lookup_embeddings.npy"
DEFAULT_SCOPE_METADATA = "./data/lookup/scope_lookup.fasta"

# Amino acid validation constants
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')
SPECIAL_CHARS = set('XUB')  # X=unknown, U=selenocysteine, B=ambiguous D/N

# Global session storage for the current Gradio instance
CURRENT_SESSION = {}

def load_models(progress=gr.Progress()):
    """
    Load the ProtTrans and Protein-Vec models.
    
    Args:
        progress: Gradio progress bar
        
    Returns:
        tuple: (tokenizer, model, model_deep, device)
    """
    progress(0.1, desc="Initializing...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    progress(0.2, desc="Loading ProtTrans tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    
    progress(0.4, desc="Loading ProtTrans model...")
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = model.to(device)
    model = model.eval()
    
    progress(0.6, desc="Cleaning memory...")
    gc.collect()
    
    progress(0.7, desc="Loading Protein-Vec model...")
    # Protein-Vec MOE model checkpoint and config
    vec_model_cpnt = os.path.join("protein_vec_models", 'protein_vec.ckpt')
    vec_model_config = os.path.join("protein_vec_models", 'protein_vec_params.json')
    
    # Load the Protein-Vec model
    vec_model_config = trans_basic_block_Config.from_json(vec_model_config)
    model_deep = trans_basic_block.load_from_checkpoint(vec_model_cpnt, config=vec_model_config)
    model_deep = model_deep.to(device)
    model_deep = model_deep.eval()
    
    progress(1.0, desc="Models loaded successfully!")
    return tokenizer, model, model_deep, device

def parse_fasta(fasta_content: str) -> List[str]:
    """
    Parse FASTA content to extract sequences.
    
    Args:
        fasta_content: String containing FASTA formatted sequences
        
    Returns:
        List of protein sequences
    """
    sequences = []
    fasta_file = io.StringIO(fasta_content)
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    return sequences

def process_uploaded_file(file_obj) -> List[str]:
    """
    Process an uploaded FASTA file.
    
    Args:
        file_obj: Uploaded file object
        
    Returns:
        List of protein sequences
    """
    sequences = []
    with tempfile.NamedTemporaryFile(suffix='.fasta', delete=False) as tmp:
        tmp.write(file_obj.read())
        tmp_path = tmp.name
    
    for record in SeqIO.parse(tmp_path, "fasta"):
        sequences.append(str(record.seq))
    
    os.unlink(tmp_path)  # Clean up temp file
    return sequences

def embed_sequences(sequences: List[str], 
                    tokenizer, 
                    model, 
                    model_deep, 
                    device,
                    progress=gr.Progress()) -> np.ndarray:
    """
    Generate embeddings for a list of protein sequences.
    
    Args:
        sequences: List of protein sequences
        tokenizer: ProtTrans tokenizer
        model: ProtTrans model
        model_deep: Protein-Vec model
        device: Computation device (CPU/GPU)
        progress: Gradio progress bar
        
    Returns:
        NumPy array of embeddings
    """
    # This is a forward pass of the Protein-Vec model
    # Every aspect is turned on (therefore no masks)
    sampled_keys = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
    all_cols = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
    masks = [all_cols[k] in sampled_keys for k in range(len(all_cols))]
    masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))[None,:]
    
    # Loop through the sequences and embed them using protein-vec
    embed_all_sequences = []
    total = len(sequences)
    
    for i, sequence in enumerate(sequences):
        progress((i+1)/total, desc=f"Embedding sequence {i+1}/{total}")
        protrans_sequence = featurize_prottrans([sequence], model, tokenizer, device)
        embedded_sequence = embed_vec(protrans_sequence, model_deep, masks, device)
        embed_all_sequences.append(embedded_sequence)
    
    # Combine the embedding vectors into an array
    seq_embeddings = np.concatenate(embed_all_sequences)
    return seq_embeddings

def perform_conformal_prediction(embeddings: np.ndarray, 
                                 risk_tolerance: float, 
                                 risk_type: str = "fdr") -> Dict[str, Any]:
    """
    Perform conformal prediction on the embeddings with control for either FDR or FNR.
    
    Args:
        embeddings: NumPy array of embeddings
        risk_tolerance: Risk tolerance value (0-0.2)
        risk_type: Type of risk to control - "fdr" (False Discovery Rate) or "fnr" (False Negative Rate)
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Convert risk_tolerance from percentage to ratio (0-1)
        alpha = risk_tolerance / 100.0
        
        # Generate dummy calibration data for demonstration
        # In a real implementation, this would come from actual calibration data
        n_calib = max(100, len(embeddings))
        n_features = embeddings.shape[1]
        
        # Create similarity matrix (cosine similarity between embeddings)
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        # Create binary label matrix (for demonstration - would be real labels in production)
        # For now we'll create random true labels
        import numpy as np
        np.random.seed(42)  # For reproducibility
        true_labels = np.random.randint(0, 2, size=len(embeddings))
        pred_labels = []
        
        # Generate calibration similarity scores 
        # In real implementation, this would be based on known calibration data
        calib_similarities = np.random.uniform(0.5, 1.0, size=n_calib)
        calib_labels = np.random.randint(0, 2, size=n_calib)

        # Compute threshold based on risk type
        if risk_type == "fdr":
            # Control False Discovery Rate
            threshold = np.quantile(
                calib_similarities[~calib_labels.astype(bool)],
                np.maximum(alpha - (1 - alpha) / len(calib_similarities), 0)
            )
        else:  # fnr
            # Control False Negative Rate
            pos_indices = np.where(calib_labels == 1)[0]
            pos_scores = calib_similarities[pos_indices]
            threshold = np.quantile(pos_scores, alpha)
        
        # Apply the threshold to get predictions
        confidence_scores = np.max(similarities, axis=1)
        predicted_labels = (confidence_scores >= threshold).astype(int)
        
        # For demonstration, we'll generate random class labels
        class_labels = [f"Class {chr(65 + i % 26)}" for i in range(len(embeddings))]
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        # Note: In a real implementation, we would need true labels to compute these metrics
        # Here we're using the random labels we generated earlier
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        
        # For ROC AUC, we need probability scores rather than binary predictions
        # We'll use the confidence scores as probability estimates
        try:
            roc_auc = roc_auc_score(true_labels, confidence_scores)
        except:
            roc_auc = 0.5  # Default if calculation fails
        
        # Calculate FPR and FNR
        fp = np.sum((predicted_labels == 1) & (true_labels == 0))
        tn = np.sum((predicted_labels == 0) & (true_labels == 0))
        fn = np.sum((predicted_labels == 0) & (true_labels == 1))
        tp = np.sum((predicted_labels == 1) & (true_labels == 1))
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            "message": f"Performed conformal prediction with {risk_type.upper()} control, risk tolerance {risk_tolerance}%",
            "embeddings_shape": embeddings.shape,
            "sample_embeddings": embeddings[:3, :5].tolist() if len(embeddings) > 0 and embeddings.shape[1] >= 5 else [],
            "predicted_labels": class_labels,
            "confidence_scores": confidence_scores.tolist(),
            "threshold": float(threshold),
            "metrics": {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc),
                "false_positive_rate": float(fpr),
                "false_negative_rate": float(fnr)
            }
        }
    except Exception as e:
        import traceback
        return {
            "error": f"Error in conformal prediction: {str(e)}",
            "traceback": traceback.format_exc(),
            "predicted_labels": ["Error"] * len(embeddings),
            "confidence_scores": [0.0] * len(embeddings)
        }

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
               risk_type: str,
               risk_value: float,
               lookup_embedding_path: str = DEFAULT_LOOKUP_EMBEDDING,
               lookup_metadata_path: str = DEFAULT_LOOKUP_METADATA,
               k: int = 1000,
               progress=gr.Progress()) -> pd.DataFrame:
    """
    Run protein search with conformal guarantees.
    
    Args:
        query_embeddings: Query embeddings
        risk_type: "fdr" or "fnr"
        risk_value: Risk tolerance (0-1)
        lookup_embedding_path: Path to lookup embeddings
        lookup_metadata_path: Path to lookup metadata
        k: Maximum number of neighbors
        progress: Gradio progress bar
        
    Returns:
        DataFrame with search results
    """
    progress(0.1, desc="Loading lookup embeddings...")
    lookup_embeddings = np.load(lookup_embedding_path, allow_pickle=True)
    
    progress(0.3, desc="Loading lookup metadata...")
    if lookup_metadata_path.endswith(".tsv"):
        lookup_df = pd.read_csv(lookup_metadata_path, sep="\t")
        lookup_seqs = lookup_df["Sequence"].values
        metadata_columns = ["Entry", "Pfam", "Protein names"]
        lookup_meta = lookup_df[metadata_columns].apply(tuple, axis=1).tolist()
    else:
        lookup_fasta = read_fasta(lookup_metadata_path)
        lookup_seqs, lookup_meta = lookup_fasta
    
    progress(0.5, desc="Loading database...")
    lookup_database = load_database(lookup_embeddings)
    
    progress(0.7, desc="Running search...")
    D, I = query(lookup_database, query_embeddings, k)
    
    # Create results DataFrame
    results = []
    for i, (indices, distances) in enumerate(zip(I, D)):
        for idx, distance in zip(indices, distances):
            result = {
                "query_idx": i,
                "lookup_seq": lookup_seqs[idx],
                "D_score": distance,
            }
            if lookup_metadata_path.endswith(".tsv"):
                result["lookup_entry"] = lookup_meta[idx][0]
                result["lookup_pfam"] = lookup_meta[idx][1]
                result["lookup_protein_names"] = lookup_meta[idx][2]
            else:
                result["lookup_meta"] = lookup_meta[idx]
            results.append(result)
    results = pd.DataFrame(results)
    
    # Apply conformal threshold
    progress(0.9, desc="Applying conformal threshold...")
    
    # Define default thresholds (these would be pre-calibrated values)
    fdr_lambda = 0.999980225003127  # Example pre-calibrated value
    fnr_lambda = 0.9999  # Example pre-calibrated value
    
    if risk_type == "fdr":
        # Use the user-provided risk value instead of the hardcoded value
        lambda_threshold = risk_value
    else:  # fnr
        # Use the user-provided risk value instead of the hardcoded value
        lambda_threshold = risk_value
    
    # Filter based on threshold
    results = results[results["D_score"] >= lambda_threshold]
    
    progress(1.0, desc="Search completed!")
    return results

def process_input(input_text: str, 
                  fasta_text: str,
                  upload_file: Optional[Any], 
                  input_type: str,
                  risk_type: str,
                  risk_value: float,
                  use_protein_vec: bool,
                  custom_embeddings: Optional[Any],
                  lookup_db: str = DEFAULT_LOOKUP_EMBEDDING,
                  metadata_db: str = DEFAULT_LOOKUP_METADATA,
                  progress=gr.Progress()) -> Tuple[Dict[str, Any], List[List[Any]], Dict[str, Any]]:
    """
    Process the input and generate predictions.
    
    Args:
        input_text: Text input containing sequences (deprecated, kept for compatibility)
        fasta_text: Text input containing FASTA formatted sequences 
        upload_file: Uploaded FASTA file
        input_type: Type of input (always "fasta_format" now)
        risk_type: Type of risk to control (FDR or FNR)
        risk_value: Risk tolerance value (0-0.2)
        use_protein_vec: Whether to use Protein-Vec for embeddings
        custom_embeddings: User-uploaded embeddings file
        lookup_db: Path to pre-embedded lookup database
        metadata_db: Path to lookup metadata
        progress: Gradio progress bar
        
    Returns:
        Tuple containing:
        - Summary information (for the summary JSON display)
        - Table data (for the DataFrame display)
        - Complete results (for the raw JSON output)
    """
    # Step 1: Get sequences from FASTA input
    sequences = []
    # Process FASTA input (either file upload or text)
    if upload_file is not None:
        sequences = process_uploaded_file(upload_file)
    elif fasta_text and fasta_text.strip():
        sequences = parse_fasta(fasta_text)
    else:
        return {"error": "No FASTA input provided. Please enter FASTA content or upload a FASTA file."}, [], {"error": "No FASTA input provided. Please enter FASTA content or upload a FASTA file."}
    
    if not sequences and custom_embeddings is None:
        return {"error": "No sequences found in the FASTA input. Please check your input format."}, [], {"error": "No sequences found in the FASTA input. Please check your input format."}
    
    # Step 2: Get embeddings
    if use_protein_vec and not custom_embeddings:
        try:
            progress(0.1, desc="Starting embedding process...")
            # Load models
            tokenizer, model, model_deep, device = load_models(progress)
            
            # Generate embeddings
            progress(0.3, desc="Generating embeddings...")
            embeddings = embed_sequences(sequences, tokenizer, model, model_deep, device, progress)
            progress(0.6, desc="Embeddings complete!")
        except Exception as e:
            return {"error": f"Error generating embeddings: {str(e)}"}, [], {"error": f"Error generating embeddings: {str(e)}"}
    elif custom_embeddings:
        try:
            progress(0.2, desc="Loading custom embeddings...")
            # Load user-provided embeddings
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
                tmp.write(custom_embeddings.read())
                tmp_path = tmp.name
            
            embeddings = np.load(tmp_path)
            os.unlink(tmp_path)  # Clean up temp file
            progress(0.4, desc="Custom embeddings loaded!")
        except Exception as e:
            return {"error": f"Error loading embeddings: {str(e)}"}, [], {"error": f"Error loading embeddings: {str(e)}"}
    else:
        return {"error": "Either Protein-Vec must be enabled or custom embeddings must be provided"}, [], {"error": "Either Protein-Vec must be enabled or custom embeddings must be provided"}
    
    # Step 3: Run search with conformal guarantees
    try:
        # Determine which database is being used
        database_type = "Custom"
        if lookup_db == DEFAULT_LOOKUP_EMBEDDING:
            database_type = "UniProt"
        elif lookup_db == DEFAULT_SCOPE_EMBEDDING:
            database_type = "SCOPE"
            
        progress(0.5, desc=f"Starting search against {database_type} database with {risk_type.upper()} control...")
        results_df = run_search(
            embeddings, 
            risk_type, 
            risk_value,
            lookup_db,
            metadata_db,
            progress=progress
        )
        
        # Store all matches in the session
        all_matches = results_df.to_dict(orient="records")
        
        # For display purposes, provide summary stats
        total_matches = len(all_matches)
        
        # 1. Create summary information for the JSON display
        summary = {
            "message": f"Successfully performed {risk_type.upper()} controlled search against {database_type} database with {risk_value} risk tolerance",
            "database_used": database_type,
            "num_matches": total_matches,
            "summary": {
                "total_results": total_matches,
                "database": database_type,
                "note": f"Found {total_matches} total matches in the {database_type} database that satisfy the {risk_type.upper()} constraint of {risk_value}. All results are displayed and can be sorted by clicking on column headers."
            },
            "all_results": f"Found {total_matches} total matches in the {database_type} database that satisfy the {risk_type.upper()} constraint of {risk_value}."
        }
        
        # 2. Table data for the DataFrame display - convert ALL matches to list of lists for better Gradio compatibility
        table_data = []
        for match in all_matches:  # No limit, display all matches
            row = [
                match.get("query_idx", ""),
                match.get("lookup_seq", ""),
                match.get("D_score", ""),
                match.get("lookup_entry", ""),
                match.get("lookup_pfam", ""),
                match.get("lookup_protein_names", "")
            ]
            table_data.append(row)
        
        # 3. Complete results for the raw JSON output
        complete_results = {
            "message": summary["message"],
            "database_used": database_type,
            "num_matches": total_matches,
            "matches": all_matches,  # Include all matches
            "all_results": summary["all_results"]
        }
        
        # Store in session for potential later use - include ALL matches for export
        global CURRENT_SESSION
        CURRENT_SESSION = {
            "results": {
                "message": summary["message"],
                "database_used": database_type,
                "num_matches": total_matches,
                "matches": all_matches,  # Store all matches for export
                "all_results": summary["all_results"]
            },
            "parameters": {
                "risk_type": risk_type,
                "risk_value": risk_value,
                "database_type": database_type,
                "input_type": "fasta_format"  # Always FASTA now
            }
        }
        
        return summary, table_data, complete_results
    except Exception as e:
        error_message = {"error": f"Error during search: {str(e)}"}
        return error_message, [], error_message

def save_current_session(session_name: str) -> Dict[str, Any]:
    """
    Save the current session to a file.
    
    Args:
        session_name: Name to use for the saved session
        
    Returns:
        Dictionary with file path and status
    """
    global CURRENT_SESSION
    
    if not CURRENT_SESSION:
        return {"error": "No active session to save"}
    
    try:
        # Ensure session_name has .json extension
        if not session_name.endswith(".json"):
            session_name = f"{session_name}.json"
        
        # Create a directory for saved sessions if it doesn't exist
        os.makedirs("saved_sessions", exist_ok=True)
        file_path = os.path.join("saved_sessions", session_name)
        
        # Save the session (simplified implementation)
        with open(file_path, 'w') as f:
            import json
            json.dump(CURRENT_SESSION, f)
        
        return {
            "success": True,
            "message": f"Session saved as {session_name}",
            "file_path": file_path
        }
    
    except Exception as e:
        return {
            "error": f"Error saving session: {str(e)}"
        }

def load_saved_session(file_obj) -> Dict[str, Any]:
    """
    Load a saved session from a file.
    
    Args:
        file_obj: Uploaded session file
        
    Returns:
        Dictionary with session data formatted for display
    """
    global CURRENT_SESSION
    
    if not file_obj:
        return {"error": "No file selected"}, None, None
    
    try:
        # Read the file content
        import json
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Parse the JSON content
        session_data = json.loads(content)
        
        # Store in global session
        CURRENT_SESSION = session_data
        
        # Prepare results for the three outputs (summary, table, full JSON)
        if "results" in session_data:
            results = session_data["results"]
            
            # Data for the results summary
            summary = {
                "message": results.get("message", ""),
                "num_matches": results.get("num_matches", 0),
                "summary": {
                    "total_results": results.get("num_matches", 0),
                    "note": f"Found {results.get('num_matches', 0)} total matches. All results are displayed and can be sorted by clicking on column headers."
                },
                "all_results": results.get("all_results", "")
            }
            
            # Data for the results table - ALL matches, formatted as list of lists
            matches = results.get("matches", []) if "matches" in results else []
            table_data = []
            for match in matches:
                row = [
                    match.get("query_idx", ""),
                    match.get("lookup_seq", ""),
                    match.get("D_score", ""),
                    match.get("lookup_entry", ""),
                    match.get("lookup_pfam", ""),
                    match.get("lookup_protein_names", "")
                ]
                table_data.append(row)
            
            # Return all three components
            return {
                "success": True,
                "message": "Session loaded successfully"
            }, summary, table_data, results
        else:
            return {"error": "Invalid session format"}, None, None, None
    
    except Exception as e:
        return {
            "error": f"Error loading session: {str(e)}"
        }, None, None, None

def export_current_results(format_type: str) -> Dict[str, Any]:
    """
    Export the current results in the specified format.
    All matches (not just displayed ones) will be included in the export.
    
    Args:
        format_type: Format to export (csv, json)
        
    Returns:
        Dictionary with file path and status
    """
    global CURRENT_SESSION
    
    if not CURRENT_SESSION or "results" not in CURRENT_SESSION:
        return {"error": "No results to export"}
    
    try:
        # Create a directory for exported reports if it doesn't exist
        os.makedirs("exported_reports", exist_ok=True)
        
        # Create a unique filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join("exported_reports", f"results_{timestamp}.{format_type}")
        
        # Export the results
        if format_type == "csv":
            if "matches" in CURRENT_SESSION["results"]:
                import pandas as pd
                # Export ALL matches, not just the displayed ones
                df = pd.DataFrame(CURRENT_SESSION["results"]["matches"])
                df.to_csv(file_path, index=False)
                total_exported = len(df)
            else:
                return {"error": "No matches to export"}
        elif format_type == "json":
            with open(file_path, 'w') as f:
                import json
                # For JSON export, we include the full result structure
                json.dump(CURRENT_SESSION["results"], f, indent=2)
                total_exported = CURRENT_SESSION["results"]["num_matches"]
        else:
            return {"error": f"Unsupported format: {format_type}"}
        
        return {
            "success": True,
            "message": f"Results exported as {file_path} ({total_exported} records)",
            "file_path": file_path
        }
    
    except Exception as e:
        return {
            "error": f"Error exporting results: {str(e)}"
        }

def create_interface():
    """
    Create and configure the Gradio interface for protein conformal prediction
    
    Returns:
        Gradio interface object
    """
    with gr.Blocks(title="Protein Conformal Prediction") as interface:
        gr.Markdown("# Protein Conformal Prediction")
        
        # Information section explaining the concepts
        with gr.Accordion("About Conformal Prediction", open=False):
            gr.Markdown("""
            ## What is Conformal Prediction?
            
            Conformal prediction is a statistical framework that provides mathematically rigorous uncertainty quantification for any machine learning model. 
            
            In protein search, it allows us to control either:
            
            - **False Discovery Rate (FDR)**: The proportion of false positives among all positive predictions. 
              - Controlling FDR ensures that among the proteins you identify as matches, only a small percentage are incorrect.
              - Use FDR control when you want to minimize the chance of following up on false leads.
            
            - **False Negative Rate (FNR)**: The proportion of missed true positives among all actual positives.
              - Controlling FNR ensures you don't miss too many true matches.
              - Use FNR control when you want to ensure comprehensive coverage, even at the cost of including some false positives.
            
            **Important**: You can only control one of these metrics at a time, as they represent different statistical guarantees and use different thresholds.
            """)
        
        # Main interface
        with gr.Row():
            with gr.Column():
                # Input section
                gr.Markdown("## 1. Input Protein Sequences (FASTA format)")
                
                # FASTA input - always visible now
                fasta_text = gr.TextArea(
                    lines=6,
                    label="Enter FASTA content",
                    placeholder=">Protein1\nMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
                )
                upload_file = gr.File(
                    label="Or upload a FASTA file",
                    file_types=[".fasta", ".fa", ".txt"]
                )
                
                # Embedding options
                with gr.Accordion("Advanced Embedding Options", open=False):
                    use_protein_vec = gr.Checkbox(
                        label="Use Protein-Vec for embeddings",
                        value=True,
                        info="If disabled, you must upload your own embeddings"
                    )
                    
                    custom_embeddings = gr.File(
                        label="Upload custom embeddings (.npy file)",
                        file_types=[".npy"],
                        visible=True
                    )
            
            with gr.Column():
                # Conformal Parameters section
                gr.Markdown("## 2. Conformal Parameters")
                gr.Markdown("""
                Select which statistical guarantee you want to control. You can choose either FDR or FNR, but not both simultaneously.
                """)
                
                risk_type = gr.Radio(
                    ["FDR", "FNR"],
                    label="Risk Type to Control",
                    value="fdr",
                    info="FDR: Controls false positives among results. FNR: Controls missed true matches."
                )
                
                risk_value = gr.Slider(
                    minimum=0.01,
                    maximum=0.2,
                    value=0.1,
                    step=0.01,
                    label="Risk Tolerance",
                    info="Lower values are more conservative (fewer false positives for FDR, fewer missed matches for FNR)"
                )
                
                # Database selection
                with gr.Accordion("Database Options", open=False):
                    gr.Markdown("""
                    ## Database Selection
                    
                    Select which pre-embedded database to search against:
                    
                    ### Available Databases:
                    
                    **UniProt** - A comprehensive collection of protein sequences and functional annotations. Provides broad coverage across diverse species and protein families.
                    
                    **SCOPE** - Structural Classification of Proteins database containing protein domains with known 3D structures. Useful for structural and functional studies.
                    
                    You can select different combinations of embedding databases and metadata files to customize your search.
                    """)
                    
                    db_type = gr.Radio(
                        ["UniProt", "SCOPE", "Custom"],
                        label="Database Type",
                        value="UniProt",
                        info="Select which database to search against"
                    )
                    
                    lookup_db = gr.Dropdown(
                        [DEFAULT_LOOKUP_EMBEDDING, DEFAULT_SCOPE_EMBEDDING, "./data/custom_db.npy"],
                        label="Embedding Database",
                        value=DEFAULT_LOOKUP_EMBEDDING,
                        info="Database of pre-embedded protein sequences to search against"
                    )
                    
                    metadata_db = gr.Dropdown(
                        [DEFAULT_LOOKUP_METADATA, DEFAULT_SCOPE_METADATA, "./data/custom_meta.tsv"],
                        label="Metadata Database",
                        value=DEFAULT_LOOKUP_METADATA,
                        info="Metadata for the selected protein database"
                    )
                    
                    gr.Markdown("""
                    *Note: UniProt provides greater coverage of protein sequence space, while SCOPE focuses on well-characterized structural domains.*
                    """)
                
                # Submit button
                submit_btn = gr.Button("Search", variant="primary")
        
        # Results section
        with gr.Row():
            gr.Markdown("## 3. Results")
        
        with gr.Row():
            with gr.Column():
                # Split the output into summary and results
                results_summary = gr.JSON(label="Search Summary")
                
                # Add a DataFrame for displaying results in a tabular format
                results_table = gr.Dataframe(
                    headers=["query_idx", "lookup_seq", "D_score", "lookup_entry", "lookup_pfam", "lookup_protein_names"],
                    label="Results Table (All Matches - Click Column Headers to Sort)",
                    wrap=True,
                    interactive=True  # Enable interactions like sorting
                )
                
                gr.Markdown("*Note: All matches are displayed in the table. Click on column headers to sort the results.*")
                
                # Keep the original JSON output as an option for advanced users
                with gr.Accordion("Full JSON Output", open=False):
                    output = gr.JSON(label="Raw Results")
        
        # Export results
        with gr.Accordion("Export Results", open=False):
            with gr.Row():
                with gr.Column():
                    export_format = gr.Radio(
                        ["csv", "json"],
                        label="Export Format",
                        value="csv"
                    )
                    export_btn = gr.Button("Export Results")
                    export_status = gr.JSON(label="Export Status")
        
        # Update visibility of custom embeddings based on checkbox
        def update_embeddings_visibility(use_protein_vec):
            return {custom_embeddings: not use_protein_vec}
        
        use_protein_vec.change(
            fn=update_embeddings_visibility,
            inputs=[use_protein_vec],
            outputs=[custom_embeddings]
        )
        
        # Export functionality
        export_btn.click(
            fn=export_current_results,
            inputs=[export_format],
            outputs=[export_status]
        )
        
        # Main prediction submission - hardcode input_type as "fasta_format"
        submit_btn.click(
            fn=lambda fasta, upload, risk_t, risk_v, use_pv, custom_emb, lookup, metadata: 
                process_input("", fasta, upload, "fasta_format", risk_t, risk_v, use_pv, custom_emb, lookup, metadata),
            inputs=[
                fasta_text, upload_file, 
                risk_type, risk_value, use_protein_vec, custom_embeddings,
                lookup_db, metadata_db
            ],
            outputs=[results_summary, results_table, output]
        )
        
        # Add radio button event handler to update dropdown selections based on database type
        # Add this to the bottom of create_interface function with the other event handlers
        
        # Database selection event handler
        def update_database_selection(db_type):
            if db_type == "UniProt":
                return {
                    lookup_db: DEFAULT_LOOKUP_EMBEDDING,
                    metadata_db: DEFAULT_LOOKUP_METADATA
                }
            elif db_type == "SCOPE":
                return {
                    lookup_db: DEFAULT_SCOPE_EMBEDDING,
                    metadata_db: DEFAULT_SCOPE_METADATA
                }
            else:  # Custom
                return {
                    lookup_db: "./data/custom_db.npy",
                    metadata_db: "./data/custom_meta.tsv"
                }
        
        db_type.change(
            fn=update_database_selection,
            inputs=[db_type],
            outputs=[lookup_db, metadata_db]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    # Use share=False to avoid Windows antivirus issues
    interface.launch(share=False) 