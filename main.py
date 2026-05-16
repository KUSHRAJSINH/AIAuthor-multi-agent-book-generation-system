"""
AIuthor — Entry point.
Run: streamlit run main.py
"""
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from frontend.app import *  # noqa: F401,F403 — re-exports the Streamlit app
