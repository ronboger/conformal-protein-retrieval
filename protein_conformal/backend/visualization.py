"""
Visualization utilities for protein conformal prediction results.

This module provides visualization tools for protein structures,
similarity networks, and statistical summaries of conformal prediction results.
"""

import os
import sys
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import py3Dmol
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Tuple


def create_structure_with_heatmap(pdb_data: str, 
                                 chain_id: str, 
                                 confidence_scores: List[float]) -> Dict[str, Any]:
    """
    Create a 3D visualization of a protein structure with conformal prediction confidence scores.
    
    Args:
        pdb_data: PDB file content as a string
        chain_id: Chain identifier
        confidence_scores: List of confidence scores for each residue
        
    Returns:
        Dictionary containing the HTML viewer and related metadata
    """
    # Prepare the viewer
    viewer = py3Dmol.view(width=800, height=500)
    viewer.addModel(pdb_data, 'pdb')
    
    # Set default visualization properties
    viewer.setStyle({'cartoon': {'color': 'gray'}})
    
    # Map confidence scores to colors
    # Higher confidence = more blue, lower confidence = more red
    min_score = min(confidence_scores)
    max_score = max(confidence_scores)
    score_range = max_score - min_score if max_score > min_score else 1.0
    
    # Create a list of residue objects with their colors based on confidence scores
    residue_list = []
    for i, score in enumerate(confidence_scores):
        # Normalize the score to 0-1 range
        norm_score = (score - min_score) / score_range
        
        # Color: blue (high confidence) to red (low confidence)
        color = get_color_gradient(norm_score)
        
        # Add to the list for visualization
        residue_list.append({
            'resi': i + 1,  # 1-indexed residue position
            'chain': chain_id,
            'color': color
        })
    
    # Apply the residue colors
    viewer.setStyle({'cartoon': {'colorfunc': residue_list}})
    
    # Final viewer setup
    viewer.zoomTo()
    viewer.spin(True)  # Enable spinning
    
    # Get the viewer as embedded HTML
    viewer_html = viewer.render()
    
    # Return the viewer along with metadata
    return {
        'html_content': viewer_html,
        'num_residues': len(confidence_scores),
        'chain_id': chain_id,
        'confidence_stats': {
            'min': min_score,
            'max': max_score,
            'mean': sum(confidence_scores) / len(confidence_scores),
            'median': sorted(confidence_scores)[len(confidence_scores) // 2]
        }
    }


def get_color_gradient(value: float) -> str:
    """
    Generate a color on a blue-white-red gradient.
    
    Args:
        value: Normalized value between 0 and 1
        
    Returns:
        Hex color string
    """
    # Blue (high confidence) to white to red (low confidence)
    if value > 0.5:
        # Blue to white gradient (high confidence)
        ratio = (value - 0.5) * 2
        r = int(255 - (ratio * 255))
        g = int(255 - (ratio * 255))
        b = 255
    else:
        # White to red gradient (low confidence)
        ratio = value * 2
        r = 255
        g = int(ratio * 255)
        b = int(ratio * 255)
        
    return f'#{r:02x}{g:02x}{b:02x}'


def create_similarity_network(embeddings: np.ndarray, 
                             labels: List[str], 
                             threshold: float = 0.7,
                             max_nodes: int = 100) -> Dict[str, Any]:
    """
    Create a force-directed graph visualization showing similarity relationships.
    
    Args:
        embeddings: Array of sequence embeddings
        labels: List of sequence labels/names
        threshold: Similarity threshold for creating edges
        max_nodes: Maximum number of nodes to display
        
    Returns:
        Dictionary with the plotly figure and related data
    """
    # Limit the number of embeddings if too many
    if len(embeddings) > max_nodes:
        # Sample nodes to display
        indices = np.random.choice(len(embeddings), max_nodes, replace=False)
        embeddings = embeddings[indices]
        labels = [labels[i] for i in indices]
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes
    for i, label in enumerate(labels):
        G.add_node(i, name=label)
    
    # Compute pairwise similarities and add edges
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)
    
    # Add edges where similarity exceeds threshold
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = similarities[i][j]
            if sim > threshold:
                G.add_edge(i, j, weight=sim)
    
    # Use a layout algorithm to position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Extract node positions
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    # Create edges as plotly scatter traces
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)  # Add None to create a break in the line
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_weights.append(edge[2]['weight'])
    
    # Create a network graph using plotly
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create nodes trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    # Set node attributes
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f"{labels[node]}: {len(adjacencies[1])} connections")
    
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Protein Similarity Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    # Return the figure and metadata
    return {
        'figure': fig.to_json(),
        'node_count': len(G.nodes()),
        'edge_count': len(G.edges()),
        'avg_connectivity': sum(node_adjacencies) / len(node_adjacencies) if node_adjacencies else 0
    }


def create_statistical_summary(confidence_scores: List[float], 
                              risk_tolerance: float,
                              labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create statistical summary visualizations for conformal prediction results.
    
    Args:
        confidence_scores: List of confidence scores
        risk_tolerance: Risk tolerance parameter used for prediction
        labels: Optional list of prediction labels
        
    Returns:
        Dictionary with encoded plot images and statistics
    """
    # Initialize the figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Distribution of confidence scores
    sns.histplot(confidence_scores, kde=True, ax=ax1)
    ax1.set_title('Distribution of Confidence Scores')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Count')
    
    # Add a vertical line at the mean
    mean_score = np.mean(confidence_scores)
    ax1.axvline(mean_score, color='red', linestyle='--', 
                label=f'Mean: {mean_score:.3f}')
    
    # Add a vertical line at the specified risk tolerance
    ax1.axvline(1 - risk_tolerance/100, color='green', linestyle='-', 
                label=f'Risk Tolerance: {risk_tolerance}%')
    
    ax1.legend()
    
    # 2. Calibration plot if labels are provided
    if labels and len(labels) == len(confidence_scores):
        unique_labels = list(set(labels))
        label_indices = {label: i for i, label in enumerate(unique_labels)}
        
        # Convert labels to numeric values
        label_values = [label_indices[label] for label in labels]
        
        # Create calibration curve
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(
            y_true=label_values, 
            y_prob=confidence_scores, 
            n_bins=10
        )
        
        # Plot the calibration curve
        ax2.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve')
        
        # Add the diagonal perfect calibration line
        ax2.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        ax2.set_title('Calibration Plot')
        ax2.set_xlabel('Mean Predicted Confidence')
        ax2.set_ylabel('Empirical Proportion')
        ax2.legend()
    else:
        # If no labels, show confidence score ranking
        sorted_scores = sorted(confidence_scores, reverse=True)
        ranks = range(1, len(sorted_scores) + 1)
        
        ax2.plot(ranks, sorted_scores, marker='o', linestyle='-', alpha=0.7)
        ax2.set_title('Confidence Score Ranking')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Confidence Score')
        ax2.set_ylim(0, 1.05)
    
    # Calculate key statistics
    stats = {
        'min': min(confidence_scores),
        'max': max(confidence_scores),
        'mean': mean_score,
        'median': np.median(confidence_scores),
        'std': np.std(confidence_scores),
        'above_threshold': sum(1 for s in confidence_scores if s >= (1 - risk_tolerance/100)),
        'below_threshold': sum(1 for s in confidence_scores if s < (1 - risk_tolerance/100)),
    }
    
    # Add coverage percentage
    stats['coverage_pct'] = (stats['above_threshold'] / len(confidence_scores)) * 100
    
    # Convert the plot to a base64 encoded image
    buffer = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    return {
        'plot_image': img_str,
        'stats': stats
    }


def format_html_report(results: Dict[str, Any]) -> str:
    """
    Generate an HTML report from the visualization results.
    
    Args:
        results: Dictionary containing visualization results
        
    Returns:
        HTML string for the report
    """
    # Generate a simple HTML report with all visualizations
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Protein Conformal Prediction Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .section { margin-bottom: 30px; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
            .header { background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .plot-container { margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Protein Conformal Prediction Report</h1>
    """
    
    # Add timestamp
    html += f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
    
    # Add input summary if available
    if 'input_summary' in results:
        html += """
        <div class="section">
            <h2>Input Summary</h2>
            <div class="card">
        """
        summary = results['input_summary']
        html += f"<p>Input type: {summary.get('input_type', 'N/A')}</p>"
        html += f"<p>Number of sequences: {summary.get('num_sequences', 0)}</p>"
        
        # Add sequence lengths if available
        if 'sequence_lengths' in summary:
            seq_lengths = summary['sequence_lengths']
            html += f"<p>Average sequence length: {sum(seq_lengths) / len(seq_lengths):.1f}</p>"
            html += f"<p>Min sequence length: {min(seq_lengths)}</p>"
            html += f"<p>Max sequence length: {max(seq_lengths)}</p>"
        
        html += """
            </div>
        </div>
        """
    
    # Add 3D structure visualization if available
    if 'structure_viz' in results:
        html += """
        <div class="section">
            <h2>3D Structure Visualization</h2>
            <div class="card">
        """
        structure = results['structure_viz']
        html += structure.get('html_content', '<p>3D visualization not available</p>')
        html += """
            </div>
        </div>
        """
    
    # Add network visualization if available
    if 'network_viz' in results:
        html += """
        <div class="section">
            <h2>Similarity Network</h2>
            <div class="card">
        """
        network = results['network_viz']
        
        # Include plotly figure if available
        if 'figure' in network:
            html += f"""
            <div id="network-plot"></div>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                var figure = {network['figure']};
                Plotly.newPlot('network-plot', figure.data, figure.layout);
            </script>
            """
        
        # Add network stats
        html += f"""
        <div class="header">Network Statistics</div>
        <table>
            <tr><th>Nodes</th><td>{network.get('node_count', 0)}</td></tr>
            <tr><th>Edges</th><td>{network.get('edge_count', 0)}</td></tr>
            <tr><th>Average Connectivity</th><td>{network.get('avg_connectivity', 0):.2f}</td></tr>
        </table>
        """
        
        html += """
            </div>
        </div>
        """
    
    # Add statistical summary if available
    if 'stats_summary' in results:
        html += """
        <div class="section">
            <h2>Statistical Summary</h2>
            <div class="card">
        """
        stats = results['stats_summary']
        
        # Include the plot image
        if 'plot_image' in stats:
            html += f"""
            <div class="plot-container">
                <img src="data:image/png;base64,{stats['plot_image']}" alt="Statistical plots" style="max-width:100%;" />
            </div>
            """
        
        # Add statistical metrics
        if 'stats' in stats:
            metrics = stats['stats']
            html += """
            <div class="header">Confidence Score Statistics</div>
            <table>
            """
            for key, value in metrics.items():
                if isinstance(value, float):
                    html += f"<tr><th>{key.replace('_', ' ').title()}</th><td>{value:.4f}</td></tr>"
                else:
                    html += f"<tr><th>{key.replace('_', ' ').title()}</th><td>{value}</td></tr>"
            html += "</table>"
        
        html += """
            </div>
        </div>
        """
    
    # Close the HTML
    html += """
    </body>
    </html>
    """
    
    return html 