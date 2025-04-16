import os
from Bio import SeqIO
from Bio.Seq import Seq

def translate_fasta_files(input_folder):
    """
    Translates all DNA FASTA files in the given folder to protein sequences.
    
    Args:
        input_folder (str): Path to the folder containing FASTA files.
    """
    for filename in os.listdir(input_folder):
        if (filename.endswith(".fasta") or filename.endswith(".fa")) and not filename.endswith("_prot.fasta"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(input_folder, filename.rsplit(".", 1)[0] + "_prot.fasta")
            print("Input path: ", input_path)
            with open(output_path, "w") as output_file:
                for record in SeqIO.parse(input_path, "fasta"):
                    protein_seq = Seq(record.seq).translate(to_stop=True)  # Stops at stop codon
                    record.seq = protein_seq
                    SeqIO.write(record, output_file, "fasta")
            
            print(f"Translated: {filename} -> {output_path}")

if __name__ == "__main__":
    folder = input("Enter the path to the folder containing FASTA files: ")
    translate_fasta_files(folder)