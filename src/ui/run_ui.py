#!/usr/bin/env python3
"""
Launch script for the Quant Factor Pipeline UI
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit UI"""
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ui_file = os.path.join(current_dir, "factor_pipeline_ui.py")
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Launch Streamlit
    print("🚀 Launching Quant Factor Pipeline UI...")
    print(f"📂 UI file: {ui_file}")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", ui_file,
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 UI stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 