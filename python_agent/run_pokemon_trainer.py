#!/usr/bin/env python3
"""
Wrapper script to run unified Pokemon Crystal trainer.

This wrapper allows running the script from the main python_agent directory
with proper import paths.
"""

import sys
import os

# Add the parent directory to Python path so we can import as a package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.pokemon_trainer import main

if __name__ == "__main__":
    main()
