#!/bin/bash
# start_app.command — Double-click in Finder to launch the Xenium DGE web interface.
# Opens Streamlit at http://localhost:8501 in your default browser.

set -e
cd "$(dirname "$0")"

# Activate the conda environment
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
    source "$HOME/mambaforge/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not found. Run install_mac.sh first."
    read -rp "Press Enter to close..."
    exit 1
fi

conda activate xenium_dge 2>/dev/null || {
    echo "ERROR: 'xenium_dge' environment not found. Run install_mac.sh first."
    read -rp "Press Enter to close..."
    exit 1
}

echo "Starting Xenium DGE web interface..."
echo "Open http://localhost:8501 if your browser does not open automatically."
echo ""
streamlit run app/app.py
