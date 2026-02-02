"""
Protein Conformal Prediction package.

Core functionality for conformal protein retrieval with FDR control.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Core utilities (always available)
from .util import (
    load_database,
    query,
    get_thresh_FDR,
    get_thresh_new_FDR,
    get_thresh_new,
    simplifed_venn_abers_prediction,
    get_sims_labels,
    read_fasta,
)

# Optional GUI components (require gradio)
try:
    from .gradio_app import main as run_gradio_app
except ImportError:
    run_gradio_app = None
