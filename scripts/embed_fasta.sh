#!/bin/bash

# Set the CUDA device
CUDA_DEVICE=3

# Specify the directory with all fasta files (default is current directory), assuming 
# all protein fasta files are named *_prot.fasta
INPUT_DIR="${1:-.}"

# Create the "emb" subfolder if it doesn't exist
EMB_DIR="$INPUT_DIR/emb"
mkdir -p "$EMB_DIR"

# Loop through all *_prot.fasta files, embed
for fasta_file in "$INPUT_DIR"/*_prot.fasta; do
    # Ensure the file exists
    if [[ -f "$fasta_file" ]]; then
        # Extract base filename without extension
        base_name=$(basename "$fasta_file" "_prot.fasta")
        
        # Set output file name inside emb/ folder
        output_file="$EMB_DIR/${base_name}_emb.npy"
        
        # Run embedding command
        echo "Processing: $fasta_file -> $output_file"
        cd /home/yangk/proteins/protein-vec/src_run
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python embed_seqs.py --input_file "$fasta_file" --output_file "$output_file"
    fi
done

echo "All files processed. Embeddings saved in $EMB_DIR."

#!/bin/bash

# Set the CUDA device
CUDA_DEVICE=3

# Specify the directory with all fasta files (default is current directory), assuming 
# all protein fasta files are named *_prot.fasta
INPUT_DIR="${1:-.}"

# Create the "emb" subfolder if it doesn't exist
EMB_DIR="$INPUT_DIR/emb"
mkdir -p "$EMB_DIR"

# Loop through all *_prot.fasta files, embed
for fasta_file in "$INPUT_DIR"/*_prot.fasta; do
    # Ensure the file exists
    if [[ -f "$fasta_file" ]]; then
        # Extract base filename without extension
        base_name=$(basename "$fasta_file" "_prot.fasta")
        
        # Set output file name inside emb/ folder
        output_file="$EMB_DIR/${base_name}_emb.npy"
        
        # Run embedding command
        echo "Processing: $fasta_file -> $output_file"
        cd /home/yangk/proteins/protein-vec/src_run
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python embed_seqs.py --input_file "$fasta_file" --output_file "$output_file"
    fi
done

echo "All files processed. Embeddings saved in $EMB_DIR."
