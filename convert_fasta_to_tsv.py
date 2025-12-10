from Bio import SeqIO
import pandas as pd

# Read the FASTA file
records = list(SeqIO.parse('data/lookup/scope_lookup.fasta', 'fasta'))

# Create a DataFrame with the expected columns
data = {
    'Entry': [record.id for record in records],
    'Sequence': [str(record.seq) for record in records],
    'Pfam': [''] * len(records),  # Placeholder
    'Protein names': [''] * len(records)  # Placeholder
}

# Create and save the DataFrame as TSV
df = pd.DataFrame(data)
df.to_csv('data/lookup_embeddings_meta_data.tsv', sep='\t', index=False)
print(f'Created TSV file with {len(df)} entries') 