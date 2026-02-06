"""
Backend module for Protein Conformal Prediction.
Contains the server-side implementation components.
"""

# Main interface components (always available)
from .gradio_interface import (
    create_interface,
    process_input,
    validate_sequence,
    highlight_sequence,
    run_search,
    parse_fasta,
    process_uploaded_file,
    export_current_results
)

# Optional: Visualization module (requires py3Dmol, networkx, seaborn)
# These are not used by the main Gradio interface
try:
    from .visualization import (
        create_structure_with_heatmap,
        create_similarity_network,
        create_statistical_summary,
        format_html_report
    )
except ImportError:
    # Visualization not available - missing optional dependencies
    create_structure_with_heatmap = None
    create_similarity_network = None
    create_statistical_summary = None
    format_html_report = None

# Optional: Collaborative API module (requires fastapi, uvicorn)
try:
    from .collaborative import (
        save_session,
        load_session,
        export_report,
        create_api_app
    )
except ImportError:
    # Collaborative features not available - missing optional dependencies
    save_session = None
    load_session = None
    export_report = None
    create_api_app = None 