import pandas as pd

df = pd.read_csv("data/jcvi_full_unknown_gene.csv")

# Columns in this file include the query id and the query amino-acid sequence
# (names may vary slightly; inspect df.columns if needed)
query_id_col = "Locus tag (accession CP002027). "
seq_col = "Amino acid sequence. RNA's are labeled xrna."

queries = df[[query_id_col, seq_col]].drop_duplicates()

with open("jcvi_full_unknown_genes.fasta", "w") as f:
    for qid, seq in queries.itertuples(index=False):
        f.write(f">{qid}\n{seq}\n")
