import torch
from model_protein_moe import trans_basic_block, trans_basic_block_Config
from utils_search import *
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict 
import faiss
from Bio import SeqIO


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Protein-Vec MOE model checkpoint and config
vec_model_cpnt = 'protein_vec_models/protein_vec.ckpt'
vec_model_config = 'protein_vec_models/protein_vec_params.json'

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

afdb_path = "../../AFDB_sequences.fasta"
# use SeqIO to parse the fasta file
afdb_sequences = []
for record in SeqIO.parse(afdb_path, "fasta"):
    afdb_sequences.append(str(record.seq))

print("Number of sequences in clustered AFDB")
print(len(afdb_sequences))

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
while i < len(afdb_sequences): 
    protrans_sequence = featurize_prottrans(flat_seqs[i:i+1], model, tokenizer, device)
    embedded_sequence = embed_vec(protrans_sequence, model_deep, masks, device)
    embed_all_sequences.append(embedded_sequence)
    i = i + 1
    if i % 5000 == 0:
        print(i)    

#Combine the embedding vectors into an array

afdb_embeddings = np.concatenate(embed_all_sequences)
# save the embeddings
np.save("afdb_embeddings.npy", afdb_embeddings)