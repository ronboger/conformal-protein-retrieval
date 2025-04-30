# Protein Conformal Prediction Tool

An advanced tool for protein analysis using conformal prediction with multimodal inputs, intelligent visualizations, and collaborative features.

## Features

### 1. Multimodal Input System

The tool supports diverse data entry methods to accommodate various user workflows:

- **Sequence Textbox**: Enter protein sequences directly with syntax highlighting and real-time validation
- **PDB Upload**: Drag-and-drop zone for protein structure files with automatic parsing
- **AlphaFold Integration**: Direct querying of AlphaFold DB through UniProt accession numbers
- **FASTA Format**: Support for FASTA-formatted input either through text input or file upload
- **Custom Embeddings**: Option to upload pre-computed embeddings for analysis

### 2. Intelligent Result Visualization

Layered visualization approaches for different user expertise levels:

- **Confidence Heatmaps**: Overlay conformal prediction scores on 3D protein structures using PyMol-powered WebGL renderer
- **Similarity Networks**: Force-directed graphs showing phylogenetic relationships of predicted homologs
- **Statistical Summary Cards**: At-a-glance metrics for FDR control effectiveness and power analysis

### 3. Collaborative Features

Tools for knowledge sharing and reproducibility:

- **Session Snapshots**: Save/load complete analysis states including parameters and results
- **Export Templates**: Generate preformatted reports in various formats (HTML, PDF, CSV, Markdown)
- **API Endpoints**: Core functionality exposed through RESTful interface for pipeline integration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/protein-conformal-prediction.git
cd protein-conformal-prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Gradio Interface

```bash
python -m protein_conformal.gradio_app
```

#### Command Line Options

- `--host`: Host to run the server on (default: 127.0.0.1)
- `--port`: Port to run the server on (default: 7860)
- `--debug`: Run in debug mode
- `--share`: Create a shareable link
- `--api`: Start the API server alongside the UI
- `--api-port`: Port to run the API server on (default: 8000)

### Using the Web Interface

1. **Input** tab: Choose your input method and enter protein sequences, upload files, or query AlphaFold.
2. **Conformal Parameters** tab: Configure risk tolerance and confidence level for the analysis.
3. **Embedding Options** tab: Select whether to use Protein-Vec or custom embeddings.
4. Click the "Run Prediction" button to perform the analysis.
5. **Visualizations** tab: Explore the 3D structures, similarity networks, and statistical summaries.
6. **Collaboration** tab: Save/load sessions, export reports, and access API information.

### Using the API

The tool provides a RESTful API for programmatic access:

```python
import requests

# Submit a prediction request
response = requests.post(
    "http://127.0.0.1:8000/predict",
    data={
        "input_type": "protein_sequence",
        "risk_tolerance": 5.0,
        "confidence_level": "95%",
        "use_protein_vec": True,
        "sequences": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYN"
    }
)

print(response.json())
```

Key endpoints:
- `/predict`: Submit prediction requests
- `/save-session`: Save a session
- `/export-report`: Export results in various formats

## File Structure

```
protein_conformal/
├── backend/
│   ├── __init__.py
│   ├── gradio_interface.py         # Basic Gradio interface
│   ├── enhanced_gradio_interface.py # Enhanced interface with visualizations
│   ├── visualization.py            # Visualization utilities
│   ├── collaborative.py            # Session management and API functionality
├── gradio_app.py                   # Main entry point
├── __init__.py
└── README.md
```

## Requirements

See `requirements.txt` for the full list of dependencies. 