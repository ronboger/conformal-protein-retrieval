from transformers import AutoModelForMaskedLM, AutoTokenizer
from protein_conformal.scope_utils import calculate_pppl
import torch

torch.cuda.set_device(0)
device = torch.device("cuda:0")

# Load the model and tokenizer
model_name = "facebook/esm1b_t33_650M_UR50S" ## this is the backbone ESM model used for CLEAN
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model = model.cuda()

# load sequences from pa][ndas dataframe "data/new.csv" under "Sequence" column
import pandas as pd

"""
df = pd.read_csv("data/new.csv", sep = "\t")
new_sequences = df["Sequence"]

# same for "data/price.csv"
df = pd.read_csv("data/price.csv", sep = "\t")
price_sequences = df["Sequence"]

# Calculate PPPL for all sequences
pppl_new = [calculate_pppl(model, tokenizer, seq, device) for seq in new_sequences]
pppl_price = [calculate_pppl(model, tokenizer, seq, device) for seq in price_sequences]

# Save the results
df["PPPL"] = pppl_price
df.to_csv("data/price_w_pppl.csv", sep = "\t", index = False)

df = pd.read_csv("data/new.csv", sep = "\t")
df["PPPL"] = pppl_new

df.to_csv("data/new_w_pppl.csv", sep = "\t", index = False)
"""

df = pd.read_csv("data/split100.csv", sep = "\t")
split100_sequences = df['Sequence']

pppl_train = [calculate_pppl(model, tokenizer, seq, device) for seq in split100_sequences]

df["PPPL"] = pppl_train
df.to_csv("data/split100_w_pppl.csv", sep="\t", index = False)
