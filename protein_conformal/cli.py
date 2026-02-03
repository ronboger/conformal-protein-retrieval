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

    CLEAN uses ESM-1b embeddings (1280-dim) passed through a LayerNormNet (128-dim).
    Requires CLEAN package: https://github.com/tttianhao/CLEAN
    """
    import numpy as np
    import torch

    try:
        from CLEAN.model import LayerNormNet
    except ImportError:
        print("Error: CLEAN package not installed.")
        print("Install from: https://github.com/tttianhao/CLEAN")
        print("  cd CLEAN_repo/app && python build.py install")
        sys.exit(1)

    # Find CLEAN pretrained weights
    repo_root = Path(__file__).parent.parent
    clean_data_dir = repo_root / "CLEAN_repo" / "app" / "data" / "pretrained"
    model_file = args.clean_model if hasattr(args, 'clean_model') and args.clean_model else "split100"

    model_path = clean_data_dir / f"{model_file}.pth"
    if not model_path.exists():
        # Try alternate location
        model_path = Path(f"./data/pretrained/{model_file}.pth")

    if not model_path.exists():
        print(f"Error: CLEAN model weights not found at {model_path}")
        print("Download pretrained weights from the CLEAN repository:")
        print("  https://drive.google.com/file/d/1kwYd4VtzYuMvJMWXy6Vks91DSUAOcKpZ/view")
        sys.exit(1)

    # Load CLEAN model (512 hidden, 128 output)
    print(f"Loading CLEAN model: {model_file}")
    dtype = torch.float32
    model = LayerNormNet(512, 128, device, dtype)
    checkpoint = torch.load(str(model_path), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Step 1: Compute ESM-1b embeddings
    print("Loading ESM-1b model for CLEAN...")
    try:
        import esm
    except ImportError:
        print("Error: fair-esm package not installed.")
        print("Install with: pip install fair-esm")
        sys.exit(1)

    esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    esm_model = esm_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    # Process sequences in batches
    print("Computing ESM-1b embeddings...")
    esm_embeddings = []
    batch_size = 4  # Adjust based on GPU memory
    truncation_length = 1022  # ESM-1b max length

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        # Prepare batch data: list of (label, sequence) tuples
        batch_data = [(f"seq_{j}", seq[:truncation_length]) for j, seq in enumerate(batch_seqs)]

        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

            # Mean pool over sequence length (excluding special tokens)
            for j, seq in enumerate(batch_strs):
                seq_len = min(len(seq), truncation_length)
                # Tokens: [CLS] seq [EOS], so take tokens 1:seq_len+1
                emb = token_representations[j, 1:seq_len + 1].mean(0)
                esm_embeddings.append(emb.cpu())

        if (i + batch_size) % 20 == 0 or i + batch_size >= len(sequences):
            print(f"  ESM embeddings: {min(i + batch_size, len(sequences))}/{len(sequences)}")

    # Stack ESM embeddings
    esm_tensor = torch.stack(esm_embeddings).to(device=device, dtype=dtype)
    print(f"ESM embeddings shape: {esm_tensor.shape}")

    # Step 2: Pass through CLEAN model
    print("Computing CLEAN embeddings...")
    with torch.no_grad():
        clean_embeddings = model(esm_tensor).cpu().numpy()

    print(f"CLEAN embeddings shape: {clean_embeddings.shape}")
    return clean_embeddings




def _get_fdr_threshold(alpha: float) -> float:
    """Look up FDR threshold from precomputed table or paper value."""
    import pandas as pd

    repo_root = Path(__file__).parent.parent
    threshold_file = repo_root / "results" / "fdr_thresholds.csv"

    # Try to load from precomputed table first
    if threshold_file.exists():
        try:
            df = pd.read_csv(threshold_file)
            # Find closest alpha in table
            if 'alpha' in df.columns and 'threshold_mean' in df.columns:
                idx = (df['alpha'] - alpha).abs().idxmin()
                return df.loc[idx, 'threshold_mean']
        except Exception:
            pass

    # Paper-verified value for α=0.1 (from 100 calibration trials)
    # See docs/REPRODUCIBILITY.md for details
    PAPER_THRESHOLD_ALPHA_0_1 = 0.999980225003127

    if abs(alpha - 0.1) < 0.001:
        return PAPER_THRESHOLD_ALPHA_0_1

    # For other alpha values, warn user and provide rough estimate
    # The threshold decreases as alpha increases (more permissive)
    print(f"  Warning: No verified threshold for α={alpha}")
    print(f"  Using interpolation from paper value (α=0.1 → λ=0.99998)")
    print(f"  For accurate thresholds, run: cpr calibrate --alpha {alpha}")

    # Rough linear interpolation based on observed pattern
    # At α=0.1, λ≈0.99998; threshold decreases ~0.00001 per 0.1 alpha increase
    estimated = PAPER_THRESHOLD_ALPHA_0_1 + (0.1 - alpha) * 0.0001
    return max(0.9998, min(0.99999, estimated))


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

    # Determine threshold from --fdr, --fnr, or --threshold
    threshold = None
    if args.no_filter:
        print("No filtering (--no-filter): returning all neighbors")
    elif args.threshold:
        threshold = args.threshold
        print(f"Using manual threshold: {threshold}")
    elif args.fnr:
        # FNR threshold (TODO: add lookup table for FNR)
        print(f"FNR control at α={args.fnr} (using approximate threshold)")
        threshold = 0.9999 - args.fnr * 0.001  # Rough approximation
        print(f"  Threshold: {threshold}")
    else:
        # Default: FDR control
        fdr_alpha = args.fdr if args.fdr else 0.1
        threshold = _get_fdr_threshold(fdr_alpha)
        print(f"FDR control at α={fdr_alpha} ({fdr_alpha*100:.0f}% FDR)")
        print(f"  Threshold: {threshold:.10f}")

    # Build results
    results = []
    n_filtered = 0
    for i in range(len(query_embeddings)):
        for j in range(args.k):
            sim = D[i, j]
            idx = I[i, j]
            # Skip placeholder results (FAISS returns -1 for non-existent neighbors)
            if idx < 0:
                continue
            if threshold is not None and sim < threshold:
                n_filtered += 1
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

    # Summary
    n_queries = len(query_embeddings)
    n_with_hits = len(results_df['query_idx'].unique()) if len(results_df) > 0 else 0
    print(f"\nResults:")
    print(f"  Queries: {n_queries}")
    print(f"  Queries with confident hits: {n_with_hits} ({n_with_hits/n_queries*100:.1f}%)")
    print(f"  Total hits: {len(results_df)}")
    if threshold:
        print(f"  Filtered out: {n_filtered} below threshold")
    print(f"Saved to {args.output}")


def cmd_find(args):
    """One-step search: FASTA → embeddings → search → results with probabilities."""
    import numpy as np
    import pandas as pd
    import tempfile
    from Bio import SeqIO
    import torch
    from protein_conformal.util import load_database, query, simplifed_venn_abers_prediction, get_sims_labels

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"=== CPR Find: FASTA to Annotated Results ===")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"FDR level: {args.fdr*100:.0f}%")
    print()

    # Step 1: Read sequences
    print(f"[1/5] Reading sequences from {args.input}...")
    sequences = []
    sequence_names = []
    for record in SeqIO.parse(args.input, "fasta"):
        sequences.append(str(record.seq))
        sequence_names.append(record.id)
    print(f"  Found {len(sequences)} sequences")

    # Step 2: Embed sequences
    print(f"\n[2/5] Computing embeddings with {args.model}...")
    if args.model == 'protein-vec':
        embeddings = _embed_protein_vec(sequences, device, args)
    elif args.model == 'clean':
        embeddings = _embed_clean(sequences, device, args)
    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Step 3: Load database
    repo_root = Path(__file__).parent.parent
    db_path = args.database if args.database else repo_root / "data" / "lookup_embeddings.npy"
    meta_path = args.database_meta if args.database_meta else repo_root / "data" / "lookup_embeddings_meta_data.tsv"

    print(f"\n[3/5] Loading database from {db_path}...")
    db_embeddings = np.load(db_path)
    print(f"  Database size: {len(db_embeddings)} proteins")

    if Path(meta_path).exists():
        if str(meta_path).endswith('.tsv'):
            db_meta = pd.read_csv(meta_path, sep='\t')
        else:
            db_meta = pd.read_csv(meta_path)
    else:
        db_meta = None
        print("  Warning: No metadata file found")

    # Determine k (10% of database or max 10000)
    k = min(max(100, len(db_embeddings) // 10), 10000)
    print(f"  Using k={k} neighbors ({k/len(db_embeddings)*100:.1f}% of database)")

    # Step 4: Search
    print(f"\n[4/5] Searching...")
    index = load_database(db_embeddings)
    D, I = query(index, embeddings, k)

    # Get threshold
    threshold = _get_fdr_threshold(args.fdr)
    print(f"  FDR threshold (α={args.fdr}): {threshold:.10f}")

    # Step 5: Build results with probabilities
    print(f"\n[5/5] Building results...")

    # Load calibration data for probabilities
    cal_path = args.calibration if args.calibration else repo_root / "data" / "pfam_new_proteins.npy"
    if Path(cal_path).exists():
        cal_data = np.load(cal_path, allow_pickle=True)
        np.random.seed(42)
        np.random.shuffle(cal_data)
        cal_subset = cal_data[:100]
        X_cal, y_cal = get_sims_labels(cal_subset, partial=False)
        X_cal = X_cal.flatten()
        y_cal = y_cal.flatten()
        compute_probs = True
    else:
        compute_probs = False
        print("  Warning: No calibration data, skipping probability computation")

    results = []
    for i in range(len(embeddings)):
        for j in range(k):
            sim = D[i, j]
            idx = I[i, j]
            if idx < 0 or sim < threshold:
                continue

            row = {
                'query_name': sequence_names[i],
                'query_idx': i,
                'match_idx': idx,
                'similarity': sim,
            }

            # Add probability if calibration available
            if compute_probs:
                p0, p1 = simplifed_venn_abers_prediction(X_cal, y_cal, sim)
                row['probability'] = (p0 + p1) / 2
                row['uncertainty'] = abs(p1 - p0)

            # Add metadata
            if db_meta is not None and idx < len(db_meta):
                for col in db_meta.columns[:5]:
                    row[f'match_{col}'] = db_meta.iloc[idx][col]

            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)

    # Summary
    n_queries = len(sequences)
    n_with_hits = len(results_df['query_idx'].unique()) if len(results_df) > 0 else 0
    print(f"\n=== Results ===")
    print(f"Queries: {n_queries}")
    print(f"Queries with confident hits: {n_with_hits} ({n_with_hits/n_queries*100:.1f}%)")
    print(f"Total confident hits: {len(results_df)}")
    print(f"Output: {args.output}")


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

    # find command (one-step: FASTA → results)
    p_find = subparsers.add_parser('find',
        help='One-step search: FASTA → embed → search → annotated results',
        description='The easiest way to use CPR. Give it a FASTA file and get annotated results.')
    p_find.add_argument('--input', '-i', required=True, help='Input FASTA file with protein sequences')
    p_find.add_argument('--output', '-o', required=True, help='Output CSV with annotated hits')
    p_find.add_argument('--model', '-m', default='protein-vec',
                        choices=['protein-vec', 'clean'],
                        help='Embedding model (default: protein-vec)')
    p_find.add_argument('--fdr', type=float, default=0.1,
                        help='False discovery rate level (default: 0.1 = 10%% FDR)')
    p_find.add_argument('--database', '-d',
                        help='Database embeddings (default: data/lookup_embeddings.npy)')
    p_find.add_argument('--database-meta',
                        help='Database metadata (default: data/lookup_embeddings_meta_data.tsv)')
    p_find.add_argument('--calibration', '-c',
                        help='Calibration data for probabilities (default: data/pfam_new_proteins.npy)')
    p_find.add_argument('--cpu', action='store_true', help='Force CPU even if GPU available')
    p_find.add_argument('--clean-model', default='split100',
                        help='CLEAN model variant (default: split100)')
    p_find.set_defaults(func=cmd_find)

    # embed command
    p_embed = subparsers.add_parser('embed', help='Embed protein sequences (step 1 of manual workflow)')
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
    p_search.add_argument('--k', type=int, default=100,
                          help='Max neighbors per query (default: 100)')
    # FDR/FNR control options
    p_search.add_argument('--fdr', type=float, default=0.1,
                          help='False discovery rate level (default: 0.1 = 10%% FDR). '
                               'Automatically looks up threshold from results/fdr_thresholds.csv')
    p_search.add_argument('--fnr', type=float,
                          help='False negative rate level (alternative to --fdr). '
                               'Use this when you want to control missed true matches.')
    p_search.add_argument('--threshold', '-t', type=float,
                          help='Manual similarity threshold (overrides --fdr/--fnr). '
                               'Use this if you have a custom threshold.')
    p_search.add_argument('--no-filter', action='store_true',
                          help='Return all neighbors without filtering (for exploration)')
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
