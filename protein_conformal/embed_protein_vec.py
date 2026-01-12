'''
python protein_conformal/embed_protein_vec.py --input_file data/inputs/rcsb_pdb_4CS4.fasta --output_file data/inputs/queries_embeddings.npy --path_to_protein_vec protein_vec_models
'''

import torch
import sys
import os
import gc
import numpy as np
from Bio import SeqIO
import argparse
from transformers import T5EncoderModel, T5Tokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='Input FASTA file with proteins')
    parser.add_argument('--path_to_protein_vec', help='Path to the directory containing Protein-Vec model files', default = "protein_vec_models")
    parser.add_argument('--output_file', help='Output file to store embeddings')
    #parser.add_argument('--method', help='ESM or TMVEC', type=str, choices=['esm','tmvec'])
    args = parser.parse_args()

    # Add the protein_vec_models directory to Python's path
    sys.path.append(args.path_to_protein_vec)
    # Now import from the model_protein_moe module
    from model_protein_moe import trans_basic_block, trans_basic_block_Config
    from utils_search import *

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    #Protein-Vec MOE model checkpoint and config
    vec_model_cpnt = os.path.join(args.path_to_protein_vec, 'protein_vec.ckpt')
    vec_model_config = os.path.join(args.path_to_protein_vec, 'protein_vec_params.json')

    #Load the ProtTrans model and ProtTrans tokenizer
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    gc.collect()

    model = model.to(device)
    model = model.eval()

    #Load the model
    vec_model_config = trans_basic_block_Config.from_json(vec_model_config)
    model_deep = trans_basic_block.load_from_checkpoint(vec_model_cpnt, config=vec_model_config)
    model_deep = model_deep.to(device)
    model_deep = model_deep.eval()

    # use SeqIO to parse the fasta file
    sequences = []
    for record in SeqIO.parse(args.input_file, "fasta"):
        sequences.append(str(record.seq))

    print("Number of sequences in fasta file")
    print(len(sequences))

    # This is a forward pass of the Protein-Vec model
    # Every aspect is turned on (therefore no masks)
    sampled_keys = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
    all_cols = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
    masks = [all_cols[k] in sampled_keys for k in range(len(all_cols))]
    masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))[None,:]

    #Loop through the sequences and embed them using protein-vec
    i = 0
    embed_all_sequences = []
    print("Starting to embed")
    while i < len(sequences): 
        protrans_sequence = featurize_prottrans(sequences[i:i+1], model, tokenizer, device)
        embedded_sequence = embed_vec(protrans_sequence, model_deep, masks, device)
        embed_all_sequences.append(embedded_sequence)
        i = i + 1
        if i % 5000 == 0:
            print(i)    

    #Combine the embedding vectors into an array

    seq_embeddings = np.concatenate(embed_all_sequences)
    # save the embeddings
    np.save(args.output_file, seq_embeddings)