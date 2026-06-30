#!/usr/bin/env python
"""
Run this file to start the gradio website (right now it runs locally) 
it imports the backend functionality and launches the interface
"""

import os
# --- Prevent a torch/faiss OpenMP runtime crash on macOS arm64. ---
# Both torch (torch/lib/libomp.dylib) and faiss (faiss/.dylibs/libomp.dylib)
# ship their own OpenMP runtime. Loading two libomp in one process segfaults
# inside __kmp_suspend_64 during faiss search. KMP_DUPLICATE_LIB_OK lets the
# second load succeed; OMP_NUM_THREADS=1 keeps the runtime single-threaded and
# stable. Must be set before anything imports torch/faiss/numpy.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
import matplotlib
matplotlib.use('Agg')
# 'Agg' is a non-interactive backend,
# it can render plots to files (like PNGs) without requiring a gui

from protein_conformal.backend.gradio_interface import create_interface

# --- De-branding: custom favicon + social meta, no Gradio chrome in the UI. ---
_FAVICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "favicon.png")
_CUSTOM_HEAD = """
<meta name="description" content="Conformal Protein Retrieval — functional protein mining with statistical guarantees.">
<meta property="og:title" content="Conformal Protein Retrieval">
<meta property="og:description" content="Functional protein mining with statistical guarantees (Boger et al., Nature Communications 2025).">
<meta property="og:type" content="website">
<meta name="twitter:title" content="Conformal Protein Retrieval">
<meta name="twitter:description" content="Functional protein mining with statistical guarantees.">
<script>
// Strip Gradio's default social/meta tags ("Gradio", "@Gradio",
// "Click to try out the app!", og:url=gradio.app) so the page reads as our app.
(function () {
  function clean() {
    document.querySelectorAll('meta').forEach(function (m) {
      var p = m.getAttribute('property') || m.getAttribute('name') || '';
      var c = (m.getAttribute('content') || '').toLowerCase();
      if (p === 'og:url' && c.indexOf('gradio.app') !== -1) m.remove();
      if (p === 'twitter:creator' && c.indexOf('gradio') !== -1) m.remove();
      if ((p === 'og:description' || p === 'twitter:description') && c.indexOf('click to try') !== -1) m.remove();
      if ((p === 'og:title' || p === 'twitter:title') && c === 'gradio') m.remove();
    });
    document.title = "Conformal Protein Retrieval";
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', clean);
  else clean();
})();
// Keyboard shortcut: Ctrl/Cmd+Enter triggers the Search button.
document.addEventListener('keydown', function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    var btn = document.querySelector('button.primary');
    if (btn) btn.click();
  }
});

// Custom hover tooltips for quick-example buttons and database options.
(function () {
  var exampleTips = {
    'example-cas9': 'CRISPR-associated nuclease; long, recognizable protein search example.',
    'example-reca': 'Bacterial DNA repair and homologous recombination protein.',
    'example-coxa': 'Cytochrome c oxidase subunit; membrane-associated enzyme example.',
    'example-insulin': 'Short human hormone sequence; quick lightweight example.',
    'example-syn30': 'Unknown genes from the Syn3.0 minimal genome; multi-query discovery example.'
  };
  var databaseTips = {
    'Swiss-Prot (540K)': 'Reviewed UniProt/Swiss-Prot proteins; curated broad-coverage functional lookup database.',
    'SCOPE': 'SCOPe structural classification database for structure/fold-oriented protein search.',
    'AFDB (Clustered)': 'Clustered AlphaFold Database representatives for broad predicted-structure coverage.',
    'Euk (74K)': 'Virome database from Nomburg et al. 2024 Nature; eukaryotic viral protein candidates.',
    'Custom': 'Upload your own Protein-Vec embeddings plus FASTA/TSV metadata.'
  };

  function tooltip() {
    var t = document.getElementById('cpr-tooltip');
    if (!t) {
      t = document.createElement('div');
      t.id = 'cpr-tooltip';
      t.style.cssText = 'position:fixed;z-index:99999;max-width:320px;padding:8px 10px;border-radius:8px;background:#2d2440;color:white;font-size:13px;line-height:1.35;box-shadow:0 6px 20px rgba(0,0,0,.25);pointer-events:none;opacity:0;transition:opacity .08s ease;';
      document.body.appendChild(t);
    }
    return t;
  }
  function showTip(text, x, y) {
    var t = tooltip();
    t.textContent = text;
    t.style.left = Math.min(x + 12, window.innerWidth - 340) + 'px';
    t.style.top = Math.min(y + 14, window.innerHeight - 80) + 'px';
    t.style.opacity = '1';
  }
  function hideTip() { tooltip().style.opacity = '0'; }

  function tipForElement(el) {
    for (var id in exampleTips) {
      var root = document.getElementById(id);
      if (root && root.contains(el)) return exampleTips[id];
    }
    var dbRoot = document.getElementById('database-radio');
    if (dbRoot && dbRoot.contains(el)) {
      var node = el;
      while (node && node !== dbRoot) {
        var txt = (node.textContent || '').replace(/\s+/g, ' ').trim();
        for (var label in databaseTips) {
          if (txt === label || (txt.includes(label) && txt.length <= label.length + 24)) {
            return databaseTips[label];
          }
        }
        node = node.parentElement;
      }
    }
    return null;
  }

  document.addEventListener('pointermove', function (e) {
    var tip = tipForElement(e.target);
    if (tip) showTip(tip, e.clientX, e.clientY);
    else hideTip();
  }, {passive: true});
  document.addEventListener('pointerleave', hideTip, {passive: true});
  document.addEventListener('scroll', hideTip, true);
})();
</script>
"""

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
        share=args.share,
        footer_links=[],          # remove "Built with Gradio" + API/settings footer links
        favicon_path=_FAVICON,    # custom tab icon instead of the Gradio logo
        head=_CUSTOM_HEAD,        # override Gradio's default social meta tags
        theme=getattr(interface, "cpr_theme", None),
        css=getattr(interface, "cpr_css", None),
        # Disable the auto-generated, documented API so the app is UI-only:
        #   - openapi_url=None -> /openapi.json is not served (no schema to script against
        #     or to power the "Use via API" docs view)
        #   - docs_url/redoc_url=None -> no Swagger/ReDoc API browser
        # The interactive UI is unaffected: it uses /config, /queue/*, and /gradio_api/call/*,
        # none of which are FastAPI documentation routes.
        app_kwargs=dict(openapi_url=None, docs_url=None, redoc_url=None),
    )

if __name__ == "__main__":
    main() 