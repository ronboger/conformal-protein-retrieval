"""
Backend module for Protein Conformal Prediction.
Contains the server-side implementation components.
"""

# Main interface components
from .gradio_interface import (
    create_interface, 
    load_models, 
    process_input,
    perform_conformal_prediction,
    validate_sequence,
    highlight_sequence,
    run_search,
    parse_fasta,
    process_uploaded_file,
    embed_sequences,
    save_current_session,
    load_saved_session,
    export_current_results
)

# Visualization and collaborative modules
from .visualization import (
    create_structure_with_heatmap,
    create_similarity_network,
    create_statistical_summary,
    format_html_report
)

from .collaborative import (
    save_session,
    load_session,
    export_report,
    create_api_app
) 