#!/usr/bin/env python
"""
Run this file to start the gradio website (right now it runs locally) 
it imports the backend functionality and launches the interface
"""

import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')
# 'Agg' is a non-interactive backend,
# it can render plots to files (like PNGs) without requiring a gui

from protein_conformal.backend.gradio_interface import create_interface

def main():
    parser = argparse.ArgumentParser(description='Protein Conformal Prediction Gradio App')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    #  enable more verbose logging or other debugging features in Gradio.
    parser.add_argument('--share', action='store_true', help='Create a shareable link')
    #  If present, Gradio will attempt to create a temporary public URL
    parser.add_argument('--api', action='store_true', help='Start the API server alongside the UI')
    # If present, the script will also start a FastAPI backend API.
    parser.add_argument('--api-port', type=int, default=8000, help='Port to run the API server on')
    
    args = parser.parse_args()  

    os.makedirs("saved_sessions", exist_ok=True)
    # this line was used for storing user session data. Not currently used.

    os.makedirs("exported_reports", exist_ok=True)
    # store exported csv files from the app
    
    interface = create_interface()
    # Calls the 'create_interface' function (imported before) to build the Gradio UI.
    # The returned value, 'interface', is a Gradio Interface object.
    
    if args.api:
    # When we run the Gradio app without the --api flag, we're already getting all the core 
    # functions. 
    # the protein embedding, conformal prediction, search, and visualization all work directly 
    # in the Gradio interface.
    # 
    # if  deploying as a standalone website on HF without integration 
    # or multi-user collaboration needs - we DO NOT need the seperate backend API server.
    #
    # Im keeping this here in case in the future we want to scale, or for whatever reason need
    # a backend API server.
        import threading
        import uvicorn
        from protein_conformal.backend.collaborative import create_api_app
        
        api_app = create_api_app()
        
        # start the api server in a separate thread
        def run_api_server():
            uvicorn.run(api_app, host=args.host, port=args.api_port)    
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()     
        print(f"API server running at http://{args.host}:{args.api_port}")

    # launch the gradio interface
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        debug=args.debug,
        share=args.share
    )

if __name__ == "__main__":
    main() 