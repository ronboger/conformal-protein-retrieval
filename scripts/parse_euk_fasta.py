#!/usr/bin/env python3
"""Parse euk.fasta into metadata TSV for the CPR Gradio app.

FASTA header format: >ProteinName__AccessionID__Organism__TaxID
Output columns: Entry, Protein names, Organism, Sequence
"""

import csv
from pathlib import Path

FASTA_PATH = Path("euk.fasta")
OUT_PATH = Path("data/euk/euk_metadata.tsv")


def parse_fasta(path):
    entries = []
    current_header = None
    current_seq_parts = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    entries.append((current_header, "".join(current_seq_parts)))
                current_header = line[1:]  # strip '>'
                current_seq_parts = []
            else:
                current_seq_parts.append(line)
        # last entry
        if current_header is not None:
            entries.append((current_header, "".join(current_seq_parts)))

    return entries


def parse_header(header):
    parts = header.split("__")
    protein_name = parts[0].replace("_", " ") if len(parts) > 0 else ""
    accession = parts[1] if len(parts) > 1 else ""
    organism = parts[2].replace("_", " ") if len(parts) > 2 else ""
    return accession, protein_name, organism


def main():
    entries = parse_fasta(FASTA_PATH)
    print(f"Parsed {len(entries)} sequences from {FASTA_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Entry", "Protein names", "Organism", "Sequence"])
        for header, seq in entries:
            accession, protein_name, organism = parse_header(header)
            writer.writerow([accession, protein_name, organism, seq])

    print(f"Wrote {len(entries) + 1} lines (including header) to {OUT_PATH}")


if __name__ == "__main__":
    main()
