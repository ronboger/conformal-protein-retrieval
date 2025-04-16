#!/bin/bash

# Define the lookup directory
LOOKUP_DIR="/home/yangk/proteins/conformal_protein_search/data"

# Navigate to the conformal-protein-retrieval directory
cd /home/yangk/proteins/conformal_protein_search/conformal-protein-retrieval || exit

# Specify the directory containing FASTA and embedding files
INPUT_DIR="${1:-.}"

# Create an output directory for the results
OUTPUT_DIR="$INPUT_DIR/output"
mkdir -p "$OUTPUT_DIR"

# Precomputed probabilities file for get_probs.py
PRECOMPUTED_PROBS="$LOOKUP_DIR/pfam_new_proteins_SVA_probs_1000_bins_200_calibration_pts.csv"

# Loop through all *_prot.fasta files
for fasta_file in "$INPUT_DIR"/*_prot.fasta; do
    if [[ -f "$fasta_file" ]]; then
        # Extract base name
        base_name=$(basename "$fasta_file" "_prot.fasta")

        # Corresponding embedding file
        emb_file="$INPUT_DIR/emb/${base_name}_emb.npy"

        # Ensure embedding file exists
        if [[ -f "$emb_file" ]]; then
            # Output CSV file for search.py results
            search_output_csv="$OUTPUT_DIR/partial_pfam_${base_name}_hits.csv"

            # Run the search script, with pfam partial FDR control as filtering threshold
            echo "Running search for: $fasta_file & $emb_file -> $search_output_csv"
            python scripts/search.py --fdr --fdr_lambda 0.9999642502418673 \
                --output "$search_output_csv" \
                --query_embedding "$emb_file" \
                --query_fasta "$fasta_file" \
                --lookup_embedding "$LOOKUP_DIR/lookup_embeddings.npy" \
                --lookup_fasta "$LOOKUP_DIR/lookup_embeddings_meta_data.fasta"

            # Ensure search output file exists before proceeding
            if [[ -f "$search_output_csv" ]]; then
                # Output CSV file for get_probs.py results
                probs_output_csv="$OUTPUT_DIR/partial_pfam_${base_name}_hits_with_probs.csv"

                # Run get_probs.py
                echo "Running get_probs for: $search_output_csv -> $probs_output_csv"
                python scripts/get_probs.py --precomputed --precomputed_path "$PRECOMPUTED_PROBS" \
                    --input "$search_output_csv" --output "$probs_output_csv" --partial
            else
                echo "Warning: Search output file $search_output_csv not found. Skipping probability computation."
            fi
        else
            echo "Warning: No matching embedding found for $fasta_file. Skipping..."
        fi
    fi
done

echo "Pipeline completed. Results saved in $OUTPUT_DIR."

