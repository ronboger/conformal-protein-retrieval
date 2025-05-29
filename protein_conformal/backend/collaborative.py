"""
Collaborative features for protein conformal prediction.

I had plans for session saving and loading, report generation and sharing,
potentially collaborative workflows.

but I dont really think these are needed for the app. 
so some code in this file isnt used (actually, none of them are used atm)
keeping it here in case in the future, we want to enable a seperate backend API server.
"""

import json
import pickle
import base64
import os
import sys
import time
from datetime import datetime
import tempfile
from typing import Dict, Any, List, Optional, Union, BinaryIO
import numpy as np
from pathlib import Path

# For API functionality
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse

# Try different import paths for visualization module
try:
    # Absolute import (when used as package)
    from protein_conformal.backend.visualization import format_html_report
except ImportError:
    try:
        # Relative import (when imported from package)
        from .visualization import format_html_report
    except ImportError:
        try:
            # Direct import (when run directly)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)
            from visualization import format_html_report
        except ImportError:
            print("Warning: Could not import visualization module for HTML reports.")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "_type": "ndarray",
                "data": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": obj.shape
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def save_session(session_data: Dict[str, Any], file_path: Optional[str] = None) -> str:
    """
    Save the current session state to a file.
    
    Args:
        session_data: Dictionary containing all session data
        file_path: Optional path to save the file, if None, auto-generated
        
    Returns:
        Path to the saved session file
    """
    # Generate a default filename if none provided
    if file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"protein_conformal_session_{timestamp}.json"
    
    # Add metadata to the session
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "type": "protein_conformal_session"
    }
    
    # Combine metadata with session data
    full_data = {
        "metadata": metadata,
        "session": session_data
    }
    
    # Handle NumPy arrays and other special types
    try:
        with open(file_path, 'w') as f:
            json.dump(full_data, f, cls=NumpyEncoder, indent=2)
    except TypeError:
        # If JSON serialization fails, use pickle (creates binary file)
        pickle_path = os.path.splitext(file_path)[0] + ".pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(full_data, f)
        file_path = pickle_path
    
    return file_path


def load_session(file_path: str) -> Dict[str, Any]:
    """
    Load a saved session from a file.
    
    Args:
        file_path: Path to the session file
        
    Returns:
        Dictionary containing the session data
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Process NumPy arrays
                def process_numpy(obj):
                    if isinstance(obj, dict) and obj.get("_type") == "ndarray":
                        return np.array(obj["data"], dtype=obj["dtype"])
                    elif isinstance(obj, dict):
                        return {k: process_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [process_numpy(item) for item in obj]
                    return obj
                
                data = process_numpy(data)
        elif file_ext == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Validate the session data
        if "metadata" not in data or "session" not in data:
            raise ValueError("Invalid session file format")
        
        return data["session"]
    
    except Exception as e:
        raise ValueError(f"Error loading session: {str(e)}")


def export_report(results: Dict[str, Any], format_type: str = "html", file_path: Optional[str] = None) -> str:
    """
    Export the results in the specified format.
    
    Args:
        results: Dictionary containing the results
        format_type: Type of export format (html, pdf, markdown, etc.)
        file_path: Optional path to save the file, if None, auto-generated
        
    Returns:
        Path to the exported file
    """
    # Generate a default filename if none provided
    if file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"protein_conformal_report_{timestamp}"
    
    # Make sure the file has the correct extension
    if not file_path.endswith(f".{format_type}"):
        file_path = f"{file_path}.{format_type}"
    
    # Create the report based on the format type
    if format_type == "html":
        report_content = format_html_report(results)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    elif format_type == "json":
        with open(file_path, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)
    
    elif format_type == "csv":
        # Save basic statistics as CSV
        import csv
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            
            # Write input summary if available
            if "input_summary" in results:
                summary = results["input_summary"]
                writer.writerow(["Input Type", summary.get("input_type", "N/A")])
                writer.writerow(["Number of Sequences", summary.get("num_sequences", 0)])
            
            # Write stats summary if available
            if "stats_summary" in results and "stats" in results["stats_summary"]:
                writer.writerow([])
                writer.writerow(["Statistics", ""])
                
                stats = results["stats_summary"]["stats"]
                for key, value in stats.items():
                    writer.writerow([key.replace("_", " ").title(), value])
    
    elif format_type == "pdf":
        # Generate PDF from HTML using a library like WeasyPrint or pdfkit
        try:
            import weasyprint
            html_content = format_html_report(results)
            pdf = weasyprint.HTML(string=html_content).write_pdf()
            
            with open(file_path, 'wb') as f:
                f.write(pdf)
        except ImportError:
            try:
                import pdfkit
                html_content = format_html_report(results)
                pdfkit.from_string(html_content, file_path)
            except ImportError:
                raise ImportError("PDF export requires WeasyPrint or pdfkit. Please install one of these packages.")
    
    elif format_type == "markdown":
        # Create a simple markdown report
        md_content = f"""# Protein Conformal Prediction Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Input Summary

"""
        if "input_summary" in results:
            summary = results["input_summary"]
            md_content += f"- Input type: {summary.get('input_type', 'N/A')}\n"
            md_content += f"- Number of sequences: {summary.get('num_sequences', 0)}\n"
            
            if "sequence_lengths" in summary:
                seq_lengths = summary["sequence_lengths"]
                md_content += f"- Average sequence length: {sum(seq_lengths) / len(seq_lengths):.1f}\n"
                md_content += f"- Min sequence length: {min(seq_lengths)}\n"
                md_content += f"- Max sequence length: {max(seq_lengths)}\n"
        
        md_content += "\n## Statistical Summary\n\n"
        
        if "stats_summary" in results and "stats" in results["stats_summary"]:
            stats = results["stats_summary"]["stats"]
            md_content += "| Metric | Value |\n"
            md_content += "| ------ | ----- |\n"
            
            for key, value in stats.items():
                if isinstance(value, float):
                    md_content += f"| {key.replace('_', ' ').title()} | {value:.4f} |\n"
                else:
                    md_content += f"| {key.replace('_', ' ').title()} | {value} |\n"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    else:
        raise ValueError(f"Unsupported export format: {format_type}")
    
    return file_path


def create_api_app(model_instance=None) -> FastAPI:
    """
    Create a FastAPI application for serving the protein conformal prediction API.
    
    Args:
        model_instance: Optional instance of the prediction model
        
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Protein Conformal Prediction API",
        description="API for protein conformal prediction services",
        version="1.0.0"
    )
    
    @app.get("/")
    async def root():
        return {
            "name": "Protein Conformal Prediction API",
            "version": "1.0.0",
            "status": "active"
        }
    
    @app.post("/predict")
    async def predict(
        sequences: Optional[str] = Form(None),
        fasta_file: Optional[UploadFile] = File(None),
        pdb_file: Optional[UploadFile] = File(None),
        uniprot_id: Optional[str] = Form(None),
        input_type: str = Form(...),
        risk_tolerance: float = Form(5.0),
        use_protein_vec: bool = Form(True),
        custom_embeddings: Optional[UploadFile] = File(None)
    ):
        # Validate input parameters
        if not any([sequences, fasta_file, pdb_file, uniprot_id, custom_embeddings]):
            raise HTTPException(
                status_code=400,
                detail="At least one input source must be provided"
            )
        
        # Placeholder for actual prediction logic (should call the model)
        # In a real implementation, this would use the shared model with the UI
        try:
            # Just a dummy response for demonstration
            response = {
                "status": "success",
                "input_type": input_type,
                "message": "API prediction endpoint functioning correctly",
                "params": {
                    "risk_tolerance": risk_tolerance,
                    "use_protein_vec": use_protein_vec
                }
            }
            
            # If using the actual model instance
            if model_instance:
                # Convert input to expected format
                from tempfile import NamedTemporaryFile
                
                # Process the input based on type
                if input_type == "protein_sequence" and sequences:
                    # Handle plain sequences
                    response["sequences"] = len(sequences.split("\n"))
                
                elif input_type == "fasta_format" and fasta_file:
                    # Handle FASTA file
                    with NamedTemporaryFile(delete=False) as temp:
                        content = await fasta_file.read()
                        temp.write(content)
                        temp.flush()
                        # Would use model_instance.process_fasta(temp.name)
                        response["fasta_file"] = fasta_file.filename
                
                elif input_type == "pdb_upload" and pdb_file:
                    # Handle PDB file
                    with NamedTemporaryFile(delete=False) as temp:
                        content = await pdb_file.read()
                        temp.write(content)
                        temp.flush()
                        # Would use model_instance.process_pdb(temp.name)
                        response["pdb_file"] = pdb_file.filename
                
                elif input_type == "alphafold" and uniprot_id:
                    # Handle AlphaFold lookup
                    # Would use model_instance.query_alphafold(uniprot_id)
                    response["uniprot_id"] = uniprot_id
            
            return JSONResponse(content=response)
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error: {str(e)}"
            )
    
    @app.post("/save-session")
    async def save_session_endpoint(
        session_data: Dict[str, Any] = Body(...),
        session_name: Optional[str] = Form(None)
    ):
        try:
            # Generate session name if not provided
            if not session_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_name = f"session_{timestamp}"
            
            # Make sure it has the right extension
            if not session_name.endswith(".json"):
                session_name = f"{session_name}.json"
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp:
                temp_path = temp.name
                json.dump(session_data, temp, cls=NumpyEncoder)
            
            # Return the path for downloading
            return {
                "status": "success",
                "message": f"Session saved as {session_name}",
                "file_path": temp_path,
                "file_name": session_name
            }
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error saving session: {str(e)}"
            )
    
    @app.get("/download-session/{file_path:path}")
    async def download_session(file_path: str, file_name: Optional[str] = None):
        try:
            if not os.path.isfile(file_path):
                raise HTTPException(
                    status_code=404,
                    detail="Session file not found"
                )
            
            if file_name is None:
                file_name = os.path.basename(file_path)
            
            return FileResponse(
                path=file_path,
                filename=file_name,
                media_type="application/json"
            )
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error downloading session: {str(e)}"
            )
    
    @app.post("/export-report")
    async def export_report_endpoint(
        results: Dict[str, Any] = Body(...),
        format_type: str = Form("html"),
        report_name: Optional[str] = Form(None)
    ):
        try:
            # Validate format type
            valid_formats = ["html", "json", "csv", "pdf", "markdown"]
            if format_type not in valid_formats:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported format. Use one of: {', '.join(valid_formats)}"
                )
            
            # Generate report name if not provided
            if not report_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_name = f"report_{timestamp}.{format_type}"
            elif not report_name.endswith(f".{format_type}"):
                report_name = f"{report_name}.{format_type}"
            
            # Create a temporary file for the report
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, report_name)
            
            # Generate the report
            export_report(results, format_type, file_path)
            
            # Return the path for downloading
            return {
                "status": "success",
                "message": f"Report exported as {report_name}",
                "file_path": file_path,
                "file_name": report_name
            }
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error exporting report: {str(e)}"
            )
    
    @app.get("/download-report/{file_path:path}")
    async def download_report(file_path: str, file_name: Optional[str] = None):
        try:
            if not os.path.isfile(file_path):
                raise HTTPException(
                    status_code=404,
                    detail="Report file not found"
                )
            
            if file_name is None:
                file_name = os.path.basename(file_path)
            
            # Determine the media type based on file extension
            extension = os.path.splitext(file_path)[1].lower()
            media_types = {
                ".html": "text/html",
                ".json": "application/json",
                ".csv": "text/csv",
                ".pdf": "application/pdf",
                ".md": "text/markdown"
            }
            
            media_type = media_types.get(extension, "application/octet-stream")
            
            return FileResponse(
                path=file_path,
                filename=file_name,
                media_type=media_type
            )
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error downloading report: {str(e)}"
            )
    
    return app 