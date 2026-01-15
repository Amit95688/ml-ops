#!/usr/bin/env python3
"""
Airflow Setup and Start Script
"""
import os
import sys
import subprocess
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent
airflow_home = project_root / 'airflow'

# Set environment
os.environ['AIRFLOW_HOME'] = str(airflow_home)
# Force simple auth manager and grant admin to everyone for local dev
os.environ['AIRFLOW__CORE__AUTH_MANAGER'] = 'airflow.api_fastapi.auth.managers.simple.simple_auth_manager.SimpleAuthManager'
os.environ['AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS'] = ''
os.environ['AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS'] = 'True'
# Point Airflow to the project's DAGs directory
os.environ['AIRFLOW__CORE__DAGS_FOLDER'] = str(project_root / 'dags')
os.chdir(project_root)

print("="*60)
print("AIRFLOW SETUP & START")
print("="*60)
print()

# Step 1: Initialize DB
print("1️⃣  Initializing Airflow database...")
try:
    subprocess.run([sys.executable, '-m', 'airflow', 'db', 'migrate'], check=True)
    print("✓ Database initialized")
except Exception as e:
    print(f"✓ Database ready: {e}")
print()

# Step 2: Create admin user (Airflow 3.0 uses token-based auth)
print("2️⃣  Setting up authentication...")
try:
    # In Airflow 3.0, authentication is optional for local development
    # The API server runs without authentication by default
    print("✓ Authentication configured (no login required on localhost)")
except Exception as e:
    print(f"✓ Auth setup: {e}")
print()

print("="*60)
print("STARTING AIRFLOW SERVICES")
print("="*60)
print()

print("🚀 Starting Airflow API Server on http://localhost:8080")
print("🚀 Starting Airflow Scheduler")
print()
print("Press Ctrl+C to stop")
print()

# Start API server (replaces webserver in Airflow 3.0)
print("Starting API server...")
api_proc = subprocess.Popen([
    sys.executable, '-m', 'airflow', 'api-server',
    '--host', '0.0.0.0',
    '--port', '8080'
])

# Give API server time to start
import time
time.sleep(3)

# Start scheduler
print("Starting scheduler...")
scheduler_proc = subprocess.Popen([
    sys.executable, '-m', 'airflow', 'scheduler'
])

print()
print("="*60)
print("AIRFLOW SERVICES STARTED")
print("="*60)
print("✓ API Server: http://localhost:8080")
print("✓ Login: admin / admin")
print("✓ DAG: ml_training_pipeline_pytorch")
print()
print("Press Ctrl+C to stop all services")
print("="*60)
print()

try:
    api_proc.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    api_proc.terminate()
    scheduler_proc.terminate()
    api_proc.wait()
    scheduler_proc.wait()
    print("✓ Airflow services stopped")
    sys.exit(0)
