#!/usr/bin/env python
"""
Launch MLflow UI after model training
Run this script after training to view results in MLflow
"""
import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent
mlruns_dir = project_root / 'mlruns'

def launch_mlflow_ui(port=5000):
    """Launch MLflow UI"""
    print("="*60)
    print("LAUNCHING MLFLOW UI")
    print("="*60)
    print(f"\nMLruns directory: {mlruns_dir}")
    print(f"Starting MLflow server on http://localhost:{port}")
    print(f"\nPress Ctrl+C to stop the server\n")
    
    os.chdir(project_root)
    
    try:
        # Start MLflow UI
        cmd = [
            sys.executable,
            '-m',
            'mlflow',
            'ui',
            '--host',
            '0.0.0.0',
            '--port',
            str(port),
            '--backend-store-uri',
            f'file:{mlruns_dir}'
        ]
        
        # Open browser after a short delay
        time.sleep(2)
        try:
            webbrowser.open(f'http://localhost:{port}')
            print(f"Opened browser at http://localhost:{port}")
        except:
            print(f"Could not open browser. Visit http://localhost:{port} manually")
        
        # Run MLflow
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\nMLflow UI stopped")
        sys.exit(0)
    except Exception as e:
        print(f"Error launching MLflow UI: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch MLflow UI')
    parser.add_argument('--port', type=int, default=5000, help='Port to run MLflow UI on')
    args = parser.parse_args()
    
    launch_mlflow_ui(port=args.port)
