#!/usr/bin/env python
"""
Entry point for launching the gradio app (src layout shim).
"""
from protein_conformal.backend.gradio_interface import create_interface

def main():
    # Keep the behavior identical to the original entrypoint
    import argparse, os
    import matplotlib
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser(description='Protein Conformal Prediction Gradio App')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--api', action='store_true')
    parser.add_argument('--api-port', type=int, default=8000)
    args = parser.parse_args()

    os.makedirs("saved_sessions", exist_ok=True)
    os.makedirs("exported_reports", exist_ok=True)

    interface = create_interface()
    interface.launch(server_name=args.host, server_port=args.port, debug=args.debug, share=args.share)
