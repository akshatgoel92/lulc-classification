import subprocess
import sys
import time
import threading

def run_fastapi():
    """Run FastAPI backend."""
    subprocess.run([sys.executable, "backend.py"])

def run_streamlit():
    """Run Streamlit frontend."""
    time.sleep(2)  # Wait for FastAPI to start
    subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend.py"])

if __name__ == "__main__":
    print("Starting Satellite Imagery Visualization App...")
    print("FastAPI backend will run on http://localhost:8000")
    print("Streamlit frontend will run on http://localhost:8501")
    
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()
    
    # Start Streamlit in main thread
    run_streamlit()