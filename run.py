#!/usr/bin/env python3
"""
Run script for Satellite Imagery Viewer
Starts both FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
import threading
import signal
import os
from pathlib import Path

def run_fastapi():
    """Run the FastAPI backend server"""
    print("🚀 Starting FastAPI backend...")
    while True:
        try:
            # Run uvicorn with the FastAPI app
            subprocess.run([
                sys.executable, "-m", "uvicorn", 
                "backend:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ FastAPI backend failed: {e}")
            print("🔄 Restarting FastAPI in 5 seconds...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("🛑 FastAPI backend stopped")
            break

def run_streamlit():
    """Run the Streamlit frontend"""
    print("🚀 Starting Streamlit frontend...")
    # Wait for FastAPI to start
    time.sleep(8)
    
    while True:
        try:
            print("🔄 Starting Streamlit server...")
            
            # Run streamlit - this will run indefinitely if successful
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                "frontend.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--server.runOnSave", "false",
                "--server.fileWatcherType", "none",
                "--browser.gatherUsageStats", "false"
            ], check=True)
            
            # If we get here, Streamlit exited (which usually means an error)
            print("⚠️ Streamlit exited unexpectedly")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Streamlit frontend failed: {e}")
        except KeyboardInterrupt:
            print("🛑 Streamlit frontend stopped")
            break
        except Exception as e:
            print(f"❌ Unexpected error in Streamlit: {e}")
            
        print("🔄 Restarting Streamlit in 10 seconds...")
        time.sleep(10)

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "fastapi", "uvicorn", "streamlit", "rasterio", "geopandas", 
        "numpy", "pandas", "matplotlib", "PIL", "shapely", "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_files():
    """Check if the required app files exist"""
    backend_file = Path("backend.py")
    frontend_file = Path("frontend.py")
    
    missing_files = []
    
    if not backend_file.exists():
        missing_files.append("backend.py")
    
    if not frontend_file.exists():
        missing_files.append("frontend.py")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Make sure you save:")
        print("   - FastAPI backend code as 'backend.py'")
        print("   - Streamlit frontend code as 'frontend.py'")
        return False
    
    print("✅ Backend and frontend files found")
    return True

def main():
    """Main function to run both servers"""
    print("🛰️ Satellite Imagery Viewer - Startup Script")
    print("=" * 50)
    
    # Check dependencies and files
    if not check_dependencies():
        sys.exit(1)
    
    if not check_files():
        sys.exit(1)
    
    print("\n🔧 Configuration:")
    print("   📡 FastAPI Backend: http://localhost:8000")
    print("   🖥️  Streamlit Frontend: http://localhost:8501")
    print("   📁 Make sure your TIFF files are accessible")
    print("   📋 Backend file: backend.py")
    print("   📋 Frontend file: frontend.py")
    print("=" * 50)
    
    # Start both servers in separate threads
    try:
        # Start FastAPI in a separate thread
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        
        # Start Streamlit in the main thread (so we can see its output)
        run_streamlit()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down both servers...")
        sys.exit(0)

if __name__ == "__main__":
    main()