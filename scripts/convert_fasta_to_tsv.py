from Bio import SeqIO
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    records = list(SeqIO.parse(args.input, 'fasta'))
    data = {
        'Entry': [record.id for record in records],
        'Sequence': [str(record.seq) for record in records],
        'Pfam': [''] * len(records),
        'Protein names': [''] * len(records)
    }
    df = pd.DataFrame(data)
    df.to_csv(args.output, sep='\t', index=False)
    print(f'Created TSV file with {len(df)} entries')

if __name__ == '__main__':
    main()
