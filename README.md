# Protein conformal retrieval

Code and notebooks from [Functional protein mining with conformal guarantees](https://www.biorxiv.org/content/10.1101/2024.06.27.601042v5) (2024). All data can be found in [our Zenodo link](https://zenodo.org/records/14272215). Results can be reproduced through executing the data preparation notebooks in each of the subdirectories before running conformal protein retrieval.

## Installation

`pip install -e .`

## Structure

- `./protein_conformal`: utility functions to creating confidence sets and assigning probabilities to any protein machine learning model for search
- `./scope`: experiments pertraining to SCOPe
- `./pfam`: notebooks demonstrating how to use our techniques to calibrate false discovery and false negative rates for different pfam classes
- `./ec`: experiments pertraining to EC number classification on uniprot
- `./data`: scripts and notebooks used to process data
- `./clean_selection`: scripts and notebooks used to process data
