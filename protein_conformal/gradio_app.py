#!/usr/bin/env python
"""
Main entry point for the Protein Conformal Prediction Gradio application.
This file imports the backend functionality and launches the Gradio interface.
"""

import argparse
import os
import sys
import matplotlib
# Use a non-interactive backend for matplotlib to avoid issues on servers without display
matplotlib.use('Agg')

from protein_conformal.backend.gradio_interface import create_interface
from protein_conformal.util import load_database, query, read_fasta, get_sims_labels
from protein_conformal.util import get_thresh_new_FDR, get_thresh_new, risk, calculate_false_negatives, simplifed_venn_abers_prediction

def main():
    parser = argparse.ArgumentParser(description='Protein Conformal Prediction Gradio App')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--share', action='store_true', help='Create a shareable link')
    parser.add_argument('--api', action='store_true', help='Start the API server alongside the UI')
    parser.add_argument('--api-port', type=int, default=8000, help='Port to run the API server on')
    
    args = parser.parse_args()
    
    os.makedirs("saved_sessions", exist_ok=True)
    os.makedirs("exported_reports", exist_ok=True)
    
    interface = create_interface()
    
    # Start API server if requested
    if args.api:
        import threading
        import uvicorn
        from protein_conformal.backend.collaborative import create_api_app
        
        # Create the API app
        api_app = create_api_app()
        
        # Start the API server in a separate thread
        def run_api_server():
            uvicorn.run(api_app, host=args.host, port=args.api_port)
        
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()
        
        print(f"API server running at http://{args.host}:{args.api_port}")
    
    # Launch the Gradio interface
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        debug=args.debug,
        share=args.share
    )

if __name__ == "__main__":
    main() 