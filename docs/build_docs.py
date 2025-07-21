#!/usr/bin/env python3
"""
Build script for Sphinx documentation.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    """Build the Sphinx documentation."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"
    
    print("Building Sphinx documentation...")
    print(f"Project root: {project_root}")
    print(f"Docs directory: {docs_dir}")
    
    # Change to docs directory
    os.chdir(docs_dir)
    
    # Clean previous builds
    print("Cleaning previous builds...")
    if (docs_dir / "_build").exists():
        shutil.rmtree(docs_dir / "_build")
    
    # Generate API documentation
    print("Generating API documentation...")
    try:
        subprocess.run([
            "sphinx-apidoc",
            "-f",  # Force overwrite
            "-o", ".",
            "../src"
        ], check=True)
        print("✓ API documentation generated")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating API docs: {e}")
        return 1
    except FileNotFoundError:
        print("✗ sphinx-apidoc not found. Install sphinx with: pip install sphinx")
        return 1
    
    # Build HTML documentation
    print("Building HTML documentation...")
    try:
        subprocess.run([
            "sphinx-build",
            "-b", "html",
            "-d", "_build/doctrees",
            ".",
            "_build/html"
        ], check=True)
        print("✓ HTML documentation built")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error building HTML docs: {e}")
        return 1
    except FileNotFoundError:
        print("✗ sphinx-build not found. Install sphinx with: pip install sphinx")
        return 1
    
    # Build PDF documentation (optional)
    print("Building PDF documentation...")
    try:
        subprocess.run([
            "sphinx-build",
            "-b", "latex",
            "-d", "_build/doctrees",
            ".",
            "_build/latex"
        ], check=True)
        
        # Try to build PDF if LaTeX is available
        latex_dir = docs_dir / "_build" / "latex"
        if latex_dir.exists():
            subprocess.run(["make", "-C", str(latex_dir), "all-pdf"], check=True)
            print("✓ PDF documentation built")
        else:
            print("⚠ PDF build skipped (LaTeX not available)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ PDF build skipped (LaTeX not available)")
    
    # Check for broken links
    print("Checking for broken links...")
    try:
        subprocess.run([
            "sphinx-build",
            "-b", "linkcheck",
            "-d", "_build/doctrees",
            ".",
            "_build/linkcheck"
        ], check=True)
        print("✓ Link check completed")
    except subprocess.CalledProcessError:
        print("⚠ Link check failed")
    
    print("\n" + "="*50)
    print("Documentation build completed!")
    print(f"HTML docs: {docs_dir}/_build/html/index.html")
    print(f"PDF docs: {docs_dir}/_build/latex/DocumentQA.pdf")
    print("="*50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 