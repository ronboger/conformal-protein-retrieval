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
from PIL import Image
import base64

# Add the protein_vec_models directory to Python's path
sys.path.append("protein_vec_models")
# allows pythoin to find and import modules from that directory
try:
    from model_protein_moe import trans_basic_block, trans_basic_block_Config
    from utils_search import featurize_prottrans, embed_vec
except ImportError:
    print("Warning: Could not import Protein-Vec models. Make sure 'protein_vec_models' directory exists.")

# Add the protein_conformal directory to path for using utility functions
from protein_conformal.util import load_database, query, read_fasta, get_sims_labels, get_thresh_new_FDR, get_thresh_new, risk, calculate_false_negatives, simplifed_venn_abers_prediction

# Add matplotlib for plotting
import matplotlib.pyplot as plt

# Pre-embedded database paths
DEFAULT_LOOKUP_EMBEDDING = "./data/lookup_embeddings.npy" # default path to the UniProt lookup embeddings 
DEFAULT_LOOKUP_METADATA = "./data/lookup_embeddings_meta_data.tsv" # default path to the UniProt lookup metadata
DEFAULT_CALIBRATION_DATA = "./results/calibration_probs.csv" # default path to the calibration data

# First, add constants for the SCOPE database files at the top of the file, near the other DEFAULT constants
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

def load_models(progress=gr.Progress()):
    """
    Load the ProtTrans and Protein-Vec models.
    
    Args:
        progress: Gradio progress bar to show the user when loading
        
    Returns:
        tuple: (tokenizer, model, model_deep, device)
               - tokenizer: The ProtTrans tokenizer.
               - model: The ProtTrans encoder model.
               - model_deep: The Protein-Vec model.
               - device: The computation device (CPU or CUDA GPU).
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

def embed_sequences(sequences: List[str], # Defines a function to generate embeddings for protein sequences.
                    tokenizer, # The ProtTrans tokenizer.
                    model, # The ProtTrans encoder model.
                    model_deep, # The Protein-Vec model.
                    device, # The computation device (CPU/GPU).
                    progress=gr.Progress()) -> np.ndarray: # Gradio progress bar. Returns a NumPy array of embeddings.
    """
    Ferforms the same function as embed_protein_vec.py

    Converts raw protein sequences into numerical embeddings (vectors) by running it through 
    ProtTrans, then further process with Protein-Vec
    
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
                                 risk_type: str = "fdr",
                                 calibration_data_path: str = DEFAULT_CALIBRATION_DATA) -> Dict[str, Any]:
    """
    Performs the same function as search.py

    Perform conformal prediction on the embeddings with control for either FDR or FNR.
    
    Args:
        embeddings: NumPy array of embeddings (from embed_sequences)
        risk_tolerance: Risk level alpha (0.01-0.2)
        risk_type: Type of risk to control - "fdr" (False Discovery Rate) or "fnr" (False Negative Rate)
        calibration_data_path: Path to calibration data CSV

    What it does:
        - Loads pre-computed calibration data from CSV
        - Calculates similarity between query embeddings and database
        - Determines confidence thresholds based on risk type (FDR/FNR)
        - Applies conformal prediction methods for statistical guarantees
        - Returns matches with confidence scores
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Load calibration data from CSV file
        try:
            cal_df = pd.read_csv(calibration_data_path)
            
            # Extract similarity scores and labels
            X_cal = cal_df['similarity'].values.reshape(-1, 1)
            
            # Use exact match probabilities by default
            # For partial matches, we'd use prob_partial_p0 and prob_partial_p1
            y_cal = np.where(cal_df['prob_exact_p1'] > 0.5, 1, 0).reshape(-1, 1)
            
            print(f"Loaded calibration data with {len(X_cal)} points from CSV")
            
        except (KeyError, IndexError, TypeError, pd.errors.EmptyDataError) as e:
            # Handle case where the CSV file is not properly formatted or empty
            print(f"Error loading calibration data from CSV: {str(e)}")
            return {
                "error": f"Failed to load calibration data: {str(e)}"
            }
        
        # Calculate similarity between query embeddings and database
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Choose appropriate threshold based on risk type (FDR or FNR)
        if risk_type.lower() == "fdr":
            # For FDR control: control proportion of false discoveries 
            # FDR = FP / (FP + TP)
            threshold = get_thresh_new_FDR(X_cal, y_cal, risk_tolerance)
            risk_description = "False Discovery Rate"
            risk_formula = "FDR = FP / (FP + TP)"
            risk_explanation = ("Controls the proportion of false discoveries (incorrect matches) "
                              "among all retrieved matches. Useful when you want to ensure most "
                              "retrieved results are correct.")
        else:  # fnr
            # For FNR control: control proportion of missed true matches
            # FNR = FN / (FN + TP)
            threshold = get_thresh_new(X_cal, y_cal, risk_tolerance)
            risk_description = "False Negative Rate"
            risk_formula = "FNR = FN / (FN + TP)"
            risk_explanation = ("Controls the proportion of missed true matches among all actual matches. "
                              "Useful when you want to minimize missing true relationships, even at the "
                              "cost of including some false positives.")
        
        # Apply the threshold to get predictions
        predicted_matches = similarity_matrix >= threshold
        
        # Calculate metrics for validation (using calibration data)
        # For FDR, use the risk function to calculate the actual FDR on calibration data
        if risk_type.lower() == "fdr":
            empirical_risk = risk(X_cal, y_cal, threshold)
        else:
            # For FNR, use the calculate_false_negatives function
            empirical_risk = calculate_false_negatives(X_cal, y_cal, threshold)
        
        # Calculate metrics on the query data
        # These are approximate since we don't have ground truth
        n_matches = np.sum(predicted_matches)
        match_rate = n_matches / (len(embeddings) * similarity_matrix.shape[1])
        
        # Calculate isotonic regression probabilities for all similarity scores
        # Using Venn-Abers method for probability calibration
        probs_p0 = []
        probs_p1 = []
        
        # Sample a subset of similarity scores for calibration (to save computation)
        n_samples = min(100, np.sum(predicted_matches))
        if n_samples > 0:
            sample_indices = np.random.choice(
                np.where(predicted_matches.flatten())[0], 
                size=n_samples, 
                replace=n_samples > len(np.where(predicted_matches.flatten())[0])
            )
            
            # Calculate calibrated probabilities using Venn-Abers
            X_cal_flat = X_cal.flatten()
            y_cal_flat = y_cal.flatten()
            
            for idx in sample_indices:
                sim_score = similarity_matrix.flatten()[idx]
                p0, p1 = simplifed_venn_abers_prediction(X_cal_flat, y_cal_flat, sim_score)
                probs_p0.append(p0)
                probs_p1.append(p1)
        
        # Create simulated matches for display (since we don't have ground truth)
        match_info = []
        for i in range(len(embeddings)):
            matches_for_query = []
            for j in range(similarity_matrix.shape[1]):
                if predicted_matches[i, j]:
                    # Use Venn-Abers calibrated probability if available
                    if probs_p0 and probs_p1:
                        prob = (probs_p0[0] + probs_p1[0]) / 2  # Use first as example
                    else:
                        prob = similarity_matrix[i, j]  # Fallback to similarity
                        
                    matches_for_query.append({
                        "index": j,
                        "similarity": float(similarity_matrix[i, j]),
                        "probability": float(prob)
                    })
            
            # Sort matches for this query by similarity
            matches_for_query.sort(key=lambda x: x["similarity"], reverse=True)
            match_info.append(matches_for_query)
        
        # Return results
        return {
            "message": f"Performed conformal prediction with {risk_description} control, risk level alpha {risk_tolerance}",
            "threshold": float(threshold),
            "risk_type": risk_type,
            "risk_description": risk_description,
            "risk_formula": risk_formula,
            "risk_explanation": risk_explanation,
            "empirical_risk": float(empirical_risk),
            "n_matches": int(n_matches),
            "match_rate": float(match_rate),
            "n_calib": len(cal_df),
            "match_info": match_info,
            "has_probability_calibration": len(probs_p0) > 0,
            "probability_calibration_method": "Venn-Abers prediction with isotonic regression"
        }
    except Exception as e:
        import traceback
        return {
            "error": f"Error in conformal prediction: {str(e)}",
            "traceback": traceback.format_exc()
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
               lookup_embedding_path: str = DEFAULT_LOOKUP_EMBEDDING,
               lookup_metadata_path: str = DEFAULT_LOOKUP_METADATA,
               threshold: float = 0.0,
               k: int = 1000,
               progress=gr.Progress()) -> pd.DataFrame:
    """
    Run protein search with a specified similarity threshold.
    
    Args:
        query_embeddings: Query embeddings
        lookup_embedding_path: Path to lookup embeddings
        lookup_metadata_path: Path to lookup metadata
        threshold: Similarity threshold from conformal prediction
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
    k = min(k, len(lookup_embeddings))
    D, I = query(lookup_database, query_embeddings, k)
    
    # Create results DataFrame
    results = []
    for i, (indices, distances) in enumerate(zip(I, D)):
        for idx, distance in zip(indices, distances):
            if distance >= threshold:  # Only include results that meet the threshold
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
    
    progress(1.0, desc="Search completed!")
    return results


"""
~~~~~~~~~~~~~~~~~~~~
BELOW ARE CODE MAINLY FOR GENERATING THE GRADIO WEBSITE USER INTERFACE   
~~~~~~~~~~~~~~~~~~~~
"""

def generate_error_curve_plot(X_cal: np.ndarray, y_cal: np.ndarray, 
                            risk_type: str, threshold: float, 
                            alpha: float) -> str:
    """
    Generate a plot of the error curve (FDR or FNR) vs threshold.
    
    Args:
        X_cal: Calibration similarities
        y_cal: Calibration labels
        risk_type: 'fdr' or 'fnr'
        threshold: Conformal threshold selected
        alpha: Risk tolerance level
        
    Returns:
        Base64-encoded PNG image of the plot
    """
    plt.figure(figsize=(8, 5))
    
    # Generate a range of thresholds spanning the calibration data
    min_sim, max_sim = X_cal.min(), X_cal.max()
    lambdas = np.linspace(min_sim, max_sim, 100)
    
    # Calculate the error rate for each threshold
    if risk_type.lower() == 'fdr':
        # Calculate FDR for each lambda
        risks = [risk(X_cal, y_cal, lam) for lam in lambdas]
        plt.plot(lambdas, risks, 'b-', label='False Discovery Rate (FDR)')
        plt.ylabel('FDR (False Discovery Rate)')
    else:
        # Calculate FNR for each lambda
        fnrs = [calculate_false_negatives(X_cal, y_cal, lam) for lam in lambdas]
        plt.plot(lambdas, fnrs, 'r-', label='False Negative Rate (FNR)')
        plt.ylabel('FNR (False Negative Rate)')
    
    # Mark the selected threshold and risk level
    plt.axvline(x=threshold, color='green', linestyle='--', label=f'Selected threshold: {threshold:.6f}')
    plt.axhline(y=alpha, color='orange', linestyle=':', label=f'Alpha: {alpha:.2f}')
    
    plt.xlabel('Similarity Threshold')
    plt.title(f'{risk_type.upper()} vs Similarity Threshold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.read()).decode('utf-8')

def base64_to_image(base64_str: Optional[str]) -> Optional[Image.Image]:
    """
    Convert a base64 string to a PIL Image object.
    
    Args:
        base64_str: Base64-encoded string of an image
        
    Returns:
        PIL Image object or None if conversion fails
    """
    if not base64_str:
        return None
    
    try:
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        print(f"Error converting base64 to image: {str(e)}")
        return None

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
                  custom_lookup_upload: Optional[Any] = None,
                  custom_metadata_upload: Optional[Any] = None,
                  progress=gr.Progress()) -> Tuple[Dict[str, Any], List[List[Any]], Dict[str, Any], Optional[Image.Image]]:
    """
    Process the input and generate predictions.
    
    Args:
        input_text: Text input containing sequences (deprecated, kept for compatibility)
        fasta_text: Text input containing FASTA formatted sequences 
        upload_file: Uploaded FASTA file
        input_type: Type of input (always "fasta_format" now)
        risk_type: Type of risk to control (FDR or FNR)
        risk_value: Risk tolerance value (0-20%)
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
        - Error curve plot image (PIL Image object)
    """
    # Step 1: Get sequences from FASTA input
    sequences = []
    # Process FASTA input (either file upload or text)
    if upload_file is not None:
        sequences = process_uploaded_file(upload_file)
    elif fasta_text and fasta_text.strip():
        sequences = parse_fasta(fasta_text)
    else:
        return {"error": "No FASTA input provided. Please enter FASTA content or upload a FASTA file."}, [], {"error": "No FASTA input provided. Please enter FASTA content or upload a FASTA file."}, None
    
    if not sequences and custom_embeddings is None:
        return {"error": "No sequences found in the FASTA input. Please check your input format."}, [], {"error": "No sequences found in the FASTA input. Please check your input format."}, None
    
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
            return {"error": f"Error generating embeddings: {str(e)}"}, [], {"error": f"Error generating embeddings: {str(e)}"}, None
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
            return {"error": f"Error loading embeddings: {str(e)}"}, [], {"error": f"Error loading embeddings: {str(e)}"}, None
    else:
        return {"error": "Either Protein-Vec must be enabled or custom embeddings must be provided"}, [], {"error": "Either Protein-Vec must be enabled or custom embeddings must be provided"}, None
    
    # Handle custom uploaded database files if present
    if custom_lookup_upload is not None:
        try:
            # Save the uploaded embedding file to a temporary location
            os.makedirs(os.path.dirname(CUSTOM_UPLOAD_EMBEDDING), exist_ok=True)
            with open(CUSTOM_UPLOAD_EMBEDDING, 'wb') as f:
                f.write(custom_lookup_upload.read())
            lookup_db = CUSTOM_UPLOAD_EMBEDDING
        except Exception as e:
            return {"error": f"Error processing custom database: {str(e)}"}, [], {"error": f"Error processing custom database: {str(e)}"}, None
    
    if custom_metadata_upload is not None:
        try:
            # Save the uploaded metadata file to a temporary location
            os.makedirs(os.path.dirname(CUSTOM_UPLOAD_METADATA), exist_ok=True)
            with open(CUSTOM_UPLOAD_METADATA, 'wb') as f:
                f.write(custom_metadata_upload.read())
            metadata_db = CUSTOM_UPLOAD_METADATA
        except Exception as e:
            return {"error": f"Error processing custom metadata: {str(e)}"}, [], {"error": f"Error processing custom metadata: {str(e)}"}, None
    
    # Step 3: Perform conformal prediction
    try:
        # Determine which database is being used
        database_type = "Custom"
        if lookup_db == DEFAULT_LOOKUP_EMBEDDING:
            database_type = "UniProt"
        elif lookup_db == DEFAULT_SCOPE_EMBEDDING:
            database_type = "SCOPE"
        
        progress(0.5, desc=f"Performing conformal prediction with {risk_type} control...")
        
        # Perform conformal prediction
        conformal_results = perform_conformal_prediction(
            embeddings,
            risk_value,
            risk_type.lower(),
            calibration_data_path=DEFAULT_CALIBRATION_DATA
        )
        
        if "error" in conformal_results:
            return conformal_results, [], conformal_results, None
        
        # After performing conformal prediction, generate the error curve visualization
        try:
            # Load calibration data for visualization from CSV
            cal_df = pd.read_csv(DEFAULT_CALIBRATION_DATA)
            
            # Extract similarity scores and labels for visualization
            X_cal = cal_df['similarity'].values.reshape(-1, 1)
            
            # Use exact match probabilities by default
            y_cal = np.where(cal_df['prob_exact_p1'] > 0.5, 1, 0).reshape(-1, 1)
            
            # Generate error curve plot
            progress(0.65, desc="Generating error curve visualization...")
            error_curve_plot = generate_error_curve_plot(
                X_cal, y_cal, 
                risk_type.lower(), 
                conformal_results["threshold"],
                risk_value
            )
            
            # Add visualization to results
            conformal_results["error_curve_plot"] = error_curve_plot
            
        except Exception as e:
            # If visualization fails, just log the error and continue
            print(f"Error generating visualization: {str(e)}")
            conformal_results["error_curve_plot"] = None
        
        # Step 4: Run the search against the database with the threshold
        progress(0.7, desc=f"Searching database with conformal threshold...")
        
        # Run search with the conformal prediction threshold
        results_df = run_search(
            embeddings,
            lookup_db,
            metadata_db,  # Use metadata_db here instead of lookup_metadata_path
            threshold=conformal_results["threshold"],
            k=1000,
            progress=progress
        )
        
        # Process the results
        all_matches = results_df.to_dict(orient="records")
        
        # If probability calibration is available, add probabilities to the matches
        if conformal_results["has_probability_calibration"]:
            # Using Venn-Abers for probability calibration
            progress(0.8, desc="Calibrating probabilities...")
            
            # Load calibration data for Venn-Abers from CSV
            cal_df = pd.read_csv(DEFAULT_CALIBRATION_DATA)
            
            # Extract similarity scores and labels for probability calibration
            X_cal = cal_df['similarity'].values.reshape(-1, 1)
            
            # Use exact match probabilities by default
            y_cal = np.where(cal_df['prob_exact_p1'] > 0.5, 1, 0).reshape(-1, 1)
            
            X_cal_flat = X_cal.flatten()
            y_cal_flat = y_cal.flatten()
            
            # Calculate probabilities for each match
            for i, match in enumerate(all_matches):
                sim_score = match["D_score"]
                p0, p1 = simplifed_venn_abers_prediction(X_cal_flat, y_cal_flat, sim_score)
                # Average the probabilities as in the paper
                all_matches[i]["probability"] = (p0 + p1) / 2
                all_matches[i]["p0"] = p0  # Store individual probabilities for inspection
                all_matches[i]["p1"] = p1
        else:
            # If no probability calibration, use similarity as a proxy
            for i, match in enumerate(all_matches):
                all_matches[i]["probability"] = match["D_score"]
        
        # Store all matches in the session
        total_matches = len(all_matches)
        
        # 1. Create summary information for the JSON display
        summary = {
            "message": conformal_results["message"],
            "database_used": database_type,
            "num_matches": total_matches,
            "threshold": conformal_results["threshold"],
            "risk_type": conformal_results["risk_type"],
            "risk_description": conformal_results["risk_description"],
            "risk_formula": conformal_results["risk_formula"], 
            "risk_explanation": conformal_results["risk_explanation"],
            "empirical_risk": conformal_results["empirical_risk"],
            "n_calib": conformal_results["n_calib"],
            "match_rate": conformal_results["match_rate"],
            "error_curve_plot": conformal_results.get("error_curve_plot")
        }
        
        # 2. Table data for the DataFrame display - convert ALL matches to list of lists for better Gradio compatibility
        table_data = []
        for match in all_matches:  # No limit, display all matches
            row = [
                match.get("query_idx", ""),
                match.get("lookup_seq", ""),
                match.get("D_score", ""),
                match.get("probability", ""),
                match.get("lookup_entry", ""),
                match.get("lookup_pfam", ""),
                match.get("lookup_protein_names", "")
            ]
            table_data.append(row)
        
        # 3. Complete results for the raw JSON output
        complete_results = {
            "conformal_prediction": conformal_results,
            "database_used": database_type,
            "num_matches": total_matches,
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
                "database_type": database_type,
                "input_type": "fasta_format",  # Always FASTA now
                "n_calib": conformal_results["n_calib"]
            }
        }
        
        # Extract the base64 image for display
        error_curve_image = conformal_results.get("error_curve_plot")
        # Convert base64 to PIL image for display
        error_curve_pil = base64_to_image(error_curve_image)
        
        progress(1.0, desc="Conformal prediction complete!")
        return summary, table_data, complete_results, error_curve_pil
    except Exception as e:
        error_message = {"error": f"Error during search: {str(e)}"}
        return error_message, [], error_message, None

# def save_current_session(session_name: str) -> Dict[str, Any]:
#     """
#     Save the current session to a file.
    
#     Args:
#         session_name: Name to use for the saved session
        
#     Returns:
#         Dictionary with file path and status
#     """
#     global CURRENT_SESSION
    
#     if not CURRENT_SESSION:
#         return {"error": "No active session to save"}
    
#     try:
#         # Ensure session_name has .json extension
#         if not session_name.endswith(".json"):
#             session_name = f"{session_name}.json"
        
#         # Create a directory for saved sessions if it doesn't exist
#         os.makedirs("saved_sessions", exist_ok=True)
#         file_path = os.path.join("saved_sessions", session_name)
        
#         # Save the session (simplified implementation)
#         with open(file_path, 'w') as f:
#             import json
#             json.dump(CURRENT_SESSION, f)
        
#         return {
#             "success": True,
#             "message": f"Session saved as {session_name}",
#             "file_path": file_path
#         }
    
#     except Exception as e:
#         return {
#             "error": f"Error saving session: {str(e)}"
#         }

# def load_saved_session(file_obj) -> Dict[str, Any]:
#     """
#     Load a saved session from a file.
    
#     Args:
#         file_obj: Uploaded session file
        
#     Returns:
#         Dictionary with session data formatted for display
#     """
#     global CURRENT_SESSION
    
#     if not file_obj:
#         return {"error": "No file selected"}, None, None, None
    
#     try:
#         # Read the file content
#         import json
#         content = file_obj.read()
#         if isinstance(content, bytes):
#             content = content.decode('utf-8')
        
#         # Parse the JSON content
#         session_data = json.loads(content)
        
#         # Store in global session
#         CURRENT_SESSION = session_data
        
#         # Prepare results for the three outputs (summary, table, full JSON)
#         if "results" in session_data:
#             results = session_data["results"]
            
#             # Data for the results summary
#             summary = {
#                 "message": results.get("message", ""),
#                 "num_matches": results.get("num_matches", 0),
#                 "summary": {
#                     "total_results": results.get("num_matches", 0),
#                     "note": f"Found {results.get('num_matches', 0)} total matches. All results are displayed and can be sorted by clicking on column headers."
#                 },
#                 "all_results": results.get("all_results", "")
#             }
            
#             # Data for the results table - ALL matches, formatted as list of lists
#             matches = results.get("matches", []) if "matches" in results else []
#             table_data = []
#             for match in matches:
#                 row = [
#                     match.get("query_idx", ""),
#                     match.get("lookup_seq", ""),
#                     match.get("D_score", ""),
#                     match.get("lookup_entry", ""),
#                     match.get("lookup_pfam", ""),
#                     match.get("lookup_protein_names", "")
#                 ]
#                 table_data.append(row)
            
#             # Return all three components
#             return {
#                 "success": True,
#                 "message": "Session loaded successfully"
#             }, summary, table_data, results
#         else:
#             return {"error": "Invalid session format"}, None, None, None
    
#     except Exception as e:
#         return {
#             "error": f"Error loading session: {str(e)}"
#         }, None, None, None

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
              - Formula: FDR = FP / (FP + TP)
              - Controlling FDR ensures that among the proteins you identify as matches, only a small percentage are incorrect.
              - Use FDR control when you want to minimize the chance of following up on false leads.
            
            - **False Negative Rate (FNR)**: The proportion of missed true positives among all actual positives.
              - Formula: FNR = FN / (FN + TP)
              - Controlling FNR ensures you don't miss too many true matches.
              - Use FNR control when you want to ensure comprehensive coverage, even at the cost of including some false positives.
            
            ### How Conformal Risk Control Works
            
            This implementation uses the conformal risk–control framework from "Functional protein mining with conformal guarantees" (Boger et al., 2025). 
            
            The key idea is to choose the smallest similarity threshold λ̂ such that the empirical risk on a held-out calibration set 
            does not exceed α (up to a small finite-sample correction):
            
            λ̂ = inf{λ: (1/n)∑ℓ(Xi,Cλ(Xi)) ≤ α - (1-α)/n}
            
            Where:
            - α is your risk tolerance
            - ℓ is the loss function (FDR or FNR)
            - Cλ(X) is the retrieval set for threshold λ
            
            This formula guarantees that the expected risk E[ℓ(X,Cλ̂(X))] ≤ α under exchangeability.
            
            ### Probability Calibration
            
            In addition to thresholding, we use the Venn-Abers method with isotonic regression to transform similarity scores 
            into calibrated probabilities, which can enrich interpretation of results.
            
            The Venn-Abers prediction method:
            1. Takes a raw similarity score and adds it to the calibration data with both positive and negative labels
            2. Fits isotonic regression models on both versions of the augmented dataset
            3. Returns two probability estimates (p0, p1) that provide guaranteed bounds on the true probability
            4. The average (p0+p1)/2 serves as our calibrated probability for downstream interpretation
            
            ### Calibration Data
            
            For this implementation, we use precomputed calibration probabilities stored in a CSV file with:
            - Similarity scores between protein pairs
            - Probability values for exact matches (prob_exact_p0, prob_exact_p1)
            - Probability values for partial matches (prob_partial_p0, prob_partial_p1)
            
            All available calibration points are used automatically for every user query to provide the most reliable
            statistical guarantees for either FDR or FNR metrics.
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
                Select which statistical guarantee you want to control and your risk level alpha.
                """)
                
                risk_type = gr.Radio(
                    ["FDR", "FNR"],
                    label="Risk Type to Control",
                    value="FDR",
                    info="FDR: Controls false positives among results. FNR: Controls missed true matches."
                )
                
                risk_value = gr.Slider(
                    minimum=0.01,
                    maximum=0.2,
                    value=0.1,
                    step=0.01,
                    label="Risk Level Alpha",
                    info="Lower values provide stronger statistical guarantees but may reduce the number of matches (for FDR) or increase database size (for FNR)"
                )
                
                # Database selection
                with gr.Accordion("Database Options", open=False):
                    gr.Markdown("""
                    ## Database Selection
                    
                    Select which pre-embedded database to search against:
                    
                    ### Available Databases:
                    
                    **UniProt** - A comprehensive collection of protein sequences and functional annotations. Provides broad coverage across diverse species and protein families.
                    
                    **SCOPE** - Structural Classification of Proteins database containing protein domains with known 3D structures. Useful for structural and functional studies.
                    
                    **Custom** - Upload your own database files. Requires two files:
                    1. Embedding file (.npy): A NumPy array of protein embeddings
                    2. Metadata file: Either a FASTA file (.fasta, .fa) or a tab-separated file (.tsv) with sequence metadata
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
                    
                    # Add custom file upload components
                    custom_lookup_upload = gr.File(
                        label="Upload Custom Embedding File (.npy)",
                        file_types=[".npy"],
                        visible=False,
                    )
                    
                    custom_metadata_upload = gr.File(
                        label="Upload Custom Metadata File (.fasta, .fa, or .tsv)",
                        file_types=[".fasta", ".fa", ".tsv"],
                        visible=False,
                    )
                
                # Submit button
                submit_btn = gr.Button("Search", variant="primary")
        
        # Results section
        with gr.Row():
            gr.Markdown("## 3. Results")
        
            with gr.Row():
                with gr.Column():
                    # Conformal prediction summary
                    results_summary = gr.JSON(label="Conformal Prediction Results")
                    
                    # Add a threshold and risk visualization
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Conformal Risk Control")
                            gr.Markdown("""
                            The chart below shows the relationship between similarity threshold values and the resulting error rates (FDR or FNR).
                            The dashed line represents your selected risk level alpha.
                            """)
                        
                    # Placeholder for visualization (will be implemented in a later update)
                    risk_plot = gr.Image(label="Error Rate vs Threshold", type="pil")
                    
                    # Add a DataFrame for displaying results in a tabular format
                    results_table = gr.Dataframe(
                        headers=["query_idx", "lookup_seq", "D_score", "probability", "lookup_entry", "lookup_pfam", "lookup_protein_names"],
                        label="Results Table (All Matches - Click Column Headers to Sort)",
                        wrap=True,
                        interactive=True  # Enable interactions like sorting
                    )
                    
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
                    
                    gr.Markdown("""
                    ### Citation
                    
                    If you use this tool in your research, please cite:
                    
                    ```
                    @article{boger2025functional,
                    title={Functional protein mining with conformal guarantees},
                    author={Boger, Ron S and Chithrananda, Seyone and Angelopoulos, Anastasios N and Yoon, Peter H and Jordan, Michael I and Doudna, Jennifer A},
                    journal={Nature Communications},
                    volume={16},
                    number={1},
                    pages={85},
                    year={2025},
                    publisher={Nature Publishing Group UK London}
                    }
                    ```
                    """)
        
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
            fn=lambda fasta, upload, risk_t, risk_v, use_pv, custom_emb, lookup, metadata, custom_lookup, custom_meta: 
                process_input("", fasta, upload, "fasta_format", risk_t, risk_v, use_pv, custom_emb, lookup, metadata, custom_lookup, custom_meta),
            inputs=[
                fasta_text, upload_file, 
                risk_type, risk_value, 
                use_protein_vec, custom_embeddings,
                lookup_db, metadata_db, custom_lookup_upload, custom_metadata_upload
            ],
            outputs=[results_summary, results_table, output, risk_plot]
        )
        
        # Database selection event handler
        def update_database_selection(db_type):
            result = {}
            if db_type == "UniProt":
                result = {
                    lookup_db: DEFAULT_LOOKUP_EMBEDDING,
                    metadata_db: DEFAULT_LOOKUP_METADATA,
                    custom_lookup_upload: gr.update(visible=False),
                    custom_metadata_upload: gr.update(visible=False)
                }
            elif db_type == "SCOPE":
                result = {
                    lookup_db: DEFAULT_SCOPE_EMBEDDING,
                    metadata_db: DEFAULT_SCOPE_METADATA,
                    custom_lookup_upload: gr.update(visible=False),
                    custom_metadata_upload: gr.update(visible=False)
                }
            else:  # Custom
                result = {
                    lookup_db: "./data/custom_db.npy",
                    metadata_db: "./data/custom_meta.tsv", 
                    custom_lookup_upload: gr.update(visible=True),
                    custom_metadata_upload: gr.update(visible=True)
                }
            return result
        
        db_type.change(
            fn=update_database_selection,
            inputs=[db_type],
            outputs=[lookup_db, metadata_db, custom_lookup_upload, custom_metadata_upload]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    # Use share=False to avoid Windows antivirus issues
    interface.launch(share=False) 