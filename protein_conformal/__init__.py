"""
Protein Conformal Prediction package.
"""

import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Easy access to main components
from .gradio_app import main as run_gradio_app
