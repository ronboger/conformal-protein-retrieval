#!/usr/bin/env python
"""
CPR - Conformal Protein Retrieval CLI

Command-line interface for functional protein mining with conformal guarantees.

Usage:
    # Embed protein sequences
    cpr embed --input sequences.fasta --output embeddings.npy

    # Search for similar proteins
    cpr search --query embeddings.npy --database lookup.npy --output results.csv

    # Convert similarity scores to calibrated probabilities
    cpr prob --input results.csv --calibration data/pfam_new_proteins.npy --output results_prob.csv

    # Calibrate FDR/FNR thresholds for a new embedding model
    cpr calibrate --calibration my_calibration_data.npy --output thresholds.csv --alpha 0.1

    # Verify paper results
    cpr verify --check syn30
"""

import argparse
import sys
from pathlib import Path


def cmd_embed(args):
    """Embed protein sequences using specified model."""
    import numpy as np
    import torch
    import gc
    from Bio import SeqIO

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    print(f"Embedding model: {args.model}")

    # Parse input sequences
    print(f"Reading sequences from {args.input}...")
    sequences = [str(record.seq) for record in SeqIO.parse(args.input, "fasta")]
    print(f"Found {len(sequences)} sequences")

    if args.model == 'protein-vec':
        embeddings = _embed_protein_vec(sequences, device, args)
    elif args.model == 'clean':
        embeddings = _embed_clean(sequences, device, args)
    else:
        print(f"Unknown model: {args.model}")
        print("Available models: protein-vec, clean")
        sys.exit(1)

    print(f"Embeddings shape: {embeddings.shape}")
    np.save(args.output, embeddings)
    print(f"Saved embeddings to {args.output}")


def _embed_protein_vec(sequences, device, args):
    """Embed using Protein-Vec model."""
    import numpy as np
    import torch
    import gc
    from transformers import T5EncoderModel, T5Tokenizer

    repo_root = Path(__file__).parent.parent
    model_path = repo_root / "protein_vec_models"
    if not model_path.exists():
        print(f"Error: Protein-Vec models not found at {model_path}")
        print("Please extract protein_vec_models.gz or download from the repository.")
        sys.exit(1)

    sys.path.insert(0, str(model_path))
    from model_protein_moe import trans_basic_block, trans_basic_block_Config
    from utils_search import featurize_prottrans, embed_vec

    # Load ProtTrans model
    print("Loading ProtTrans T5 model...")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    gc.collect()
    model = model.to(device).eval()

    # Load Protein-Vec model
    print("Loading Protein-Vec model...")
    vec_model_cpnt = model_path / "protein_vec.ckpt"
    vec_model_config = model_path / "protein_vec_params.json"
    config = trans_basic_block_Config.from_json(str(vec_model_config))
    model_deep = trans_basic_block.load_from_checkpoint(str(vec_model_cpnt), config=config)
    model_deep = model_deep.to(device).eval()

    # Embedding masks (all aspects enabled)
    sampled_keys = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
    all_cols = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
    masks = [all_cols[k] in sampled_keys for k in range(len(all_cols))]
    masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))[None, :]

    # Embed sequences
    print("Embedding sequences...")
    embeddings = []
    for i, seq in enumerate(sequences):
        protrans_seq = featurize_prottrans([seq], model, tokenizer, device)
        emb = embed_vec(protrans_seq, model_deep, masks, device)
        embeddings.append(emb)
        if (i + 1) % 10 == 0 or i == len(sequences) - 1:
            print(f"  Processed {i + 1}/{len(sequences)}")

    return np.concatenate(embeddings)


def _embed_clean(sequences, device, args):
    """Embed using CLEAN model (for enzyme classification).

    Requires CLEAN package: https://github.com/tttianhao/CLEAN
    """
    import numpy as np

    try:
        from CLEAN.utils import get_ec_id_dict
        from CLEAN.model import LayerNormNet
        import torch
    except ImportError:
        print("Error: CLEAN package not installed.")
        print("Install from: https://github.com/tttianhao/CLEAN")
        print("  git clone https://github.com/tttianhao/CLEAN.git")
        print("  cd CLEAN && python setup.py install")
        sys.exit(1)

    # Load CLEAN model
    model_file = args.clean_model or "split100"
    print(f"Loading CLEAN model: {model_file}")

    dtype = torch.float32
    model = LayerNormNet(512, 128, device, dtype)

    try:
        checkpoint = torch.load(f'./data/pretrained/{model_file}.pth', map_location=device)
        model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Error: CLEAN model weights not found at ./data/pretrained/{model_file}.pth")
        print("Download pretrained weights from the CLEAN repository.")
        sys.exit(1)

    model.eval()

    # CLEAN uses ESM embeddings as input
    print("Computing ESM embeddings for CLEAN...")
    esm_embeddings = _embed_esm(sequences, device, args)

    # Pass through CLEAN model
    print("Computing CLEAN embeddings...")
    with torch.no_grad():
        esm_tensor = torch.tensor(esm_embeddings, dtype=dtype, device=device)
        clean_embeddings = model(esm_tensor).cpu().numpy()

    return clean_embeddings




def cmd_search(args):
    """Search for similar proteins using FAISS with conformal guarantees."""
    import numpy as np
    import pandas as pd
    from protein_conformal.util import load_database, query, read_fasta

    print(f"Loading query embeddings from {args.query}...")
    query_embeddings = np.load(args.query)
    print(f"  Shape: {query_embeddings.shape}")

    print(f"Loading database embeddings from {args.database}...")
    db_embeddings = np.load(args.database)
    print(f"  Shape: {db_embeddings.shape}")

    # Load metadata if provided
    if args.database_meta:
        print(f"Loading database metadata from {args.database_meta}...")
        if args.database_meta.endswith('.tsv'):
            db_meta = pd.read_csv(args.database_meta, sep='\t')
        else:
            db_meta = pd.read_csv(args.database_meta)
    else:
        db_meta = None

    # Build FAISS index and query
    print("Building FAISS index...")
    index = load_database(db_embeddings)

    print(f"Querying for top {args.k} neighbors...")
    D, I = query(index, query_embeddings, args.k)

    # Apply threshold if specified
    if args.threshold:
        print(f"Applying similarity threshold: {args.threshold}")

    # Build results
    results = []
    for i in range(len(query_embeddings)):
        for j in range(args.k):
            sim = D[i, j]
            idx = I[i, j]
            if args.threshold and sim < args.threshold:
                continue
            row = {
                'query_idx': i,
                'match_idx': idx,
                'similarity': sim,
            }
            if db_meta is not None and idx < len(db_meta):
                for col in db_meta.columns[:5]:  # First 5 metadata columns
                    row[f'match_{col}'] = db_meta.iloc[idx][col]
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"Saved {len(results_df)} results to {args.output}")


def cmd_verify(args):
    """Verify paper results."""
    import subprocess

    repo_root = Path(__file__).parent.parent

    if args.check == 'syn30':
        script = repo_root / "scripts" / "verify_syn30.py"
        print("Running JCVI Syn3.0 verification (Paper Figure 2A)...")
    elif args.check == 'fdr':
        script = repo_root / "scripts" / "verify_fdr_algorithm.py"
        print("Running FDR algorithm verification...")
    elif args.check == 'dali':
        script = repo_root / "scripts" / "verify_dali.py"
        print("Running DALI prefiltering verification (Paper Tables 4-6)...")
    elif args.check == 'clean':
        script = repo_root / "scripts" / "verify_clean.py"
        print("Running CLEAN enzyme classification verification (Paper Tables 1-2)...")
    else:
        print(f"Unknown check: {args.check}")
        print("Available checks: syn30, fdr, dali, clean")
        sys.exit(1)

    subprocess.run([sys.executable, str(script)], check=True)


def cmd_prob(args):
    """Convert similarity scores to calibrated probabilities using Venn-Abers."""
    import numpy as np
    import pandas as pd
    from protein_conformal.util import simplifed_venn_abers_prediction, get_sims_labels

    print(f"Loading calibration data from {args.calibration}...")
    cal_data = np.load(args.calibration, allow_pickle=True)

    # Prepare calibration data
    n_calib = min(args.n_calib, len(cal_data))
    np.random.seed(args.seed)
    np.random.shuffle(cal_data)
    cal_subset = cal_data[:n_calib]

    X_cal, y_cal = get_sims_labels(cal_subset, partial=False)
    X_cal = X_cal.flatten()
    y_cal = y_cal.flatten()
    print(f"  Using {n_calib} calibration samples ({len(X_cal)} pairs)")

    # Load input scores
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
        scores = df[args.score_column].values
    else:
        scores = np.load(args.input)
        if scores.ndim > 1:
            scores = scores.flatten()

    print(f"Computing probabilities for {len(scores)} scores...")

    # Compute Venn-Abers probabilities
    probs = []
    uncertainties = []
    for i, score in enumerate(scores):
        p0, p1 = simplifed_venn_abers_prediction(X_cal, y_cal, score)
        prob = (p0 + p1) / 2  # Point estimate
        uncertainty = abs(p1 - p0)
        probs.append(prob)
        uncertainties.append(uncertainty)
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(scores)}")

    # Output results
    results = pd.DataFrame({
        'score': scores,
        'probability': probs,
        'uncertainty': uncertainties,
    })

    # If input was CSV, merge with original
    if args.input.endswith('.csv'):
        for col in ['probability', 'uncertainty']:
            df[col] = results[col]
        df.to_csv(args.output, index=False)
    else:
        results.to_csv(args.output, index=False)

    print(f"Saved probabilities to {args.output}")
    print(f"  Mean probability: {np.mean(probs):.4f}")
    print(f"  Mean uncertainty: {np.mean(uncertainties):.4f}")


def cmd_calibrate(args):
    """Compute FDR/FNR thresholds from calibration data.

    This allows calibrating thresholds for a new embedding model by providing
    paired similarity scores and labels.
    """
    import numpy as np
    import pandas as pd
    from protein_conformal.util import (
        get_thresh_FDR, get_thresh_new_FDR, get_thresh_new, get_sims_labels
    )

    print(f"Loading calibration data from {args.calibration}...")
    cal_data = np.load(args.calibration, allow_pickle=True)

    n_trials = args.n_trials
    n_calib = args.n_calib
    alpha = args.alpha

    print(f"Running {n_trials} calibration trials at α={alpha}...")

    results = {
        'trial': [],
        'alpha': [],
        'fdr_threshold': [],
        'fdr_risk': [],
        'fnr_threshold': [],
    }

    for trial in range(n_trials):
        np.random.seed(args.seed + trial)
        np.random.shuffle(cal_data)
        cal_subset = cal_data[:n_calib]

        sims, labels = get_sims_labels(cal_subset, partial=False)

        # FDR threshold (Learn-then-Test)
        if args.method == 'ltt':
            lhat_fdr, risk_fdr = get_thresh_FDR(
                labels.flatten(), sims.flatten(),
                alpha=alpha, delta=args.delta, N=args.n_lambdas
            )
        else:
            # Simple quantile-based
            lhat_fdr = get_thresh_new_FDR(sims, labels, alpha)
            risk_fdr = 0.0

        # FNR threshold
        lhat_fnr = get_thresh_new(sims, labels, alpha)

        results['trial'].append(trial)
        results['alpha'].append(alpha)
        results['fdr_threshold'].append(lhat_fdr)
        results['fdr_risk'].append(risk_fdr)
        results['fnr_threshold'].append(lhat_fnr)

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{n_trials}: FDR λ={lhat_fdr:.8f}, FNR λ={lhat_fnr:.8f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)

    # Summary statistics
    print(f"\nCalibration Results (α={alpha}):")
    print(f"  FDR threshold: {results_df['fdr_threshold'].mean():.10f} ± {results_df['fdr_threshold'].std():.10f}")
    print(f"  FNR threshold: {results_df['fnr_threshold'].mean():.10f} ± {results_df['fnr_threshold'].std():.10f}")
    print(f"Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog='cpr',
        description='Conformal Protein Retrieval - Functional protein mining with statistical guarantees',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # embed command
    p_embed = subparsers.add_parser('embed', help='Embed protein sequences')
    p_embed.add_argument('--input', '-i', required=True, help='Input FASTA file')
    p_embed.add_argument('--output', '-o', required=True, help='Output .npy file for embeddings')
    p_embed.add_argument('--model', '-m', default='protein-vec',
                         choices=['protein-vec', 'clean'],
                         help='Embedding model (default: protein-vec)')
    p_embed.add_argument('--cpu', action='store_true', help='Force CPU even if GPU available')
    p_embed.add_argument('--clean-model', default='split100',
                         help='CLEAN model variant (default: split100)')
    p_embed.set_defaults(func=cmd_embed)

    # search command
    p_search = subparsers.add_parser('search', help='Search for similar proteins')
    p_search.add_argument('--query', '-q', required=True, help='Query embeddings (.npy)')
    p_search.add_argument('--database', '-d', required=True, help='Database embeddings (.npy)')
    p_search.add_argument('--database-meta', '-m', help='Database metadata (.tsv or .csv)')
    p_search.add_argument('--output', '-o', required=True, help='Output results (.csv)')
    p_search.add_argument('--k', type=int, default=10, help='Number of neighbors (default: 10)')
    p_search.add_argument('--threshold', '-t', type=float, help='Similarity threshold (e.g., 0.99998 for FDR α=0.1)')
    p_search.set_defaults(func=cmd_search)

    # verify command
    p_verify = subparsers.add_parser('verify', help='Verify paper results')
    p_verify.add_argument('--check', '-c', required=True, choices=['syn30', 'fdr', 'dali', 'clean'],
                          help='Which verification to run')
    p_verify.set_defaults(func=cmd_verify)

    # prob command - convert scores to probabilities
    p_prob = subparsers.add_parser('prob', help='Convert similarity scores to calibrated probabilities')
    p_prob.add_argument('--input', '-i', required=True,
                        help='Input scores (.npy or .csv with score column)')
    p_prob.add_argument('--calibration', '-c', required=True,
                        help='Calibration data (.npy, e.g., pfam_new_proteins.npy)')
    p_prob.add_argument('--output', '-o', required=True, help='Output CSV with probabilities')
    p_prob.add_argument('--score-column', default='similarity',
                        help='Column name for scores if input is CSV (default: similarity)')
    p_prob.add_argument('--n-calib', type=int, default=100,
                        help='Number of calibration samples to use (default: 100)')
    p_prob.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    p_prob.set_defaults(func=cmd_prob)

    # calibrate command - compute thresholds for new model
    p_calib = subparsers.add_parser('calibrate', help='Compute FDR/FNR thresholds for a new embedding model')
    p_calib.add_argument('--calibration', '-c', required=True,
                         help='Calibration data (.npy with similarity/label pairs)')
    p_calib.add_argument('--output', '-o', required=True, help='Output CSV with thresholds')
    p_calib.add_argument('--alpha', '-a', type=float, default=0.1,
                         help='Target FDR/FNR level (default: 0.1)')
    p_calib.add_argument('--n-trials', type=int, default=100,
                         help='Number of calibration trials (default: 100)')
    p_calib.add_argument('--n-calib', type=int, default=1000,
                         help='Calibration samples per trial (default: 1000)')
    p_calib.add_argument('--n-lambdas', type=int, default=5000,
                         help='Lambda grid size for LTT (default: 5000)')
    p_calib.add_argument('--delta', type=float, default=0.5,
                         help='P-value threshold for LTT (default: 0.5)')
    p_calib.add_argument('--method', choices=['ltt', 'quantile'], default='quantile',
                         help='Calibration method: ltt (Learn-then-Test) or quantile (default: quantile)')
    p_calib.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    p_calib.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
