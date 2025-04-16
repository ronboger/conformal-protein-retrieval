# Protein conformal retrieval

Code and notebooks from [Functional protein mining with conformal guarantees](https://rdcu.be/d5pJG) (2024). All data can be found in [our Zenodo link](https://zenodo.org/records/14272215). Results can be reproduced through executing the data preparation notebooks in each of the subdirectories before running conformal protein retrieval.

## Installation

### Clone the repository, install dependancies:
```
git clone https://github.com/ronboger/conformal-protein-retrieval.git
cd conformal-protein-retrieval
`pip install -e .`
```

## Structure

- `./protein_conformal`: utility functions to creating confidence sets and assigning probabilities to any protein machine learning model for search
- `./scope`: experiments pertraining to SCOPe
- `./pfam`: notebooks demonstrating how to use our techniques to calibrate false discovery and false negative rates for different pfam classes
- `./ec`: experiments pertraining to EC number classification on uniprot
- `./data`: scripts and notebooks used to process data
- `./clean_selection`: scripts and notebooks used to process data

## Getting started

After cloning + running the installation steps, you can use our scripts out of the box for calibrated search and generating probabilities of exact or partial hits against Pfam/EC domains, as well as for custom datasets utilizing other models beyond Protein-Vec/Foldseek. If searching using the Pfam calibration data to control FNR/FDR rates, download `pfam_new_proteins.npy` from the Zenodo link above.


### Creating calibration datasets 
To create your own calibration dataset for search and scoring hits with Venn-Abers probabilities, we provide an example notebook for how we create our Pfam dataset with Protein-Vec embeddings. This code should work for any arbitrary embeddings from popular models for search (ex: ESM, Evo, gLM2, TM-Vec, ProTrek, etc). This notebook can be found in `./data/create_pfam_data.ipynb'`. We provide a script to embed your query and lookup databases with Protein-Vec as well, `./protein_conformal/embed_protein_vec.py`, which can then be used to create calibration datasets for Pfam domain search. 

Note: Make sure that your calibration dataset of protein sequences and annotations is outside the training dataset of your embedding model!

### Running search using a calibrated dataset

```
# Example: search with viral domains of unknown function with FDR control of 10% (exact matches) against Pfam
python scripts/search.py \
    --fdr \
    --fdr_lambda 0.99996425 \
    --output ./data/partial_pfam_viral_hits.csv \
    --query_embedding ../protein-vec/src_run/viral_domains.npy \
    --query_fasta ../protein-vec/src_run/viral_domains.fasta \
    --lookup_embedding ./data/lookup_embeddings.npy \
    --lookup_fasta ./data/lookup_embeddings_meta_data.tsv
```

Where each of the flags are described as follows:
```
--fdr: use FDR risk control (pass one of --fdr or --fnr, not both)
--fnr: use FNR risk control 
--fdr_lambda: If precomputed a FDR lambda (embedding similarity threshold), pass here
--fnr_lambda: If precomputed a FNR lambda (embedding similarity threshold), pass here
--k: Maximimal number of neighbours to keep with FAISS per query (default of 1000 nearest neighbours)
--save_inter: save FAISS similarity scores and indicies, before running conformal-protein-retrieval
--alpha: alpha value for the calibration algorithm
--num_trails: If running calibration here, number of trials to run risk control for (randomly shuffling the calibration and test sets), default is 100.
--n_calib: number of calibration datapoints
--delta: delta value for the algorithm (default: 0.5)
--output: output CSV for the results
--add_date: add date to the output filename.
--query_embedding: query file with the embeddings (.npy format)
--query_fasta: input file containing the query sequences and metadata
--lookup_embedding: lookup file with the embeddings (.npy format)
--lookup_fasta: input file containing the lookup sequences and metadata.
```

### Generating probabilities for exact/partial functional matches.

Given a calibration dataset with similarities and binary labels indicating exact/partial matches, we provide a script to use simplified Venn-Abers/isotonic regression to get a probability for ach hit based on the embedding similarity.

```
python scripts/precompute_SVA_probs.py \
    --cal_data ./data/pfam_new_proteins.npy \  # Path to calibration data
    --output ./data/pfam_sims_to_probs.csv \  # Path to save similarity-probabilities mapping
    --partial \                              # Flag to also generate probability of partial hit
    --n_bins 1000 \                          # Number of bins for linspace between min, max similarity scores
    --n_calib 100                            # Number of calibration datapoints to use
```

### Indexing against similarity-score bins to get probabilities of exact/partial matches.

Given a dataframe containing columns of the form `{similarity, prob_exact_p0, prob_exact_p1, prob_partial_p0, prob_partial_p1}`, we can utilize it to compute probabilities for new embedding searches given a dataframe of query-lookup similarity scores:

```
python scripts/get_probs.py \
    --precomputed \                               # Use precomputed similarity-to-probability mappings
    --precomputed_path ./data/pfam_sims_to_probs.csv \  # Path to the precomputed probabilities
    --input ./data/results_no_probs.csv \         # Input dataframe with similarity scores and query-lookup metadata
    --output ./data/results_with_probs.csv \      # Output dataframe with added probability columns
    --partial                                     # Include probabilities for partial hits
```

## Requests for new features

If there are certain features/models you'd like to see expanded support/guidance for, please raise an issue with details of the i) model, and ii) search tasks you're looking to apply this work towards. We look forward to hearing from you!

## Citing our work

We'd appreciate if you cite our paper if you have used these models, notebooks, or examples for your own embedding/search tasks. The BibTex is available below:

```
@article{boger2024functional,
  title={Functional protein mining with conformal guarantees},
  author={Boger, Ron S and Chithrananda, Seyone and Angelopoulos, Anastasios N and Yoon, Peter H and Jordan, Michael I and Doudna, Jennifer A},
  journal={Nature Communications},
  year={2025},
  publisher={Nature Publishing Group}
}
```
