import subprocess
import time
import sys
import os

def launch():
    print("--- Initializing Institutional Trading System v2 ---")
    
    # Ensure we are in the root directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # 1. Start API (FastAPI)
    print("[API] Starting Dashboard API (Port 8000)...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.dashboard_api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # 2. Start Scheduler
    print("[SCHEDULER] Starting Institutional Scheduler...")
    scheduler_proc = subprocess.Popen(
        [sys.executable, "-m", "backend.scheduler"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print("\n[SUCCESS] System Launched Successfully!")
    print("Dashboard API: http://localhost:8000")
    print("Frontend: Run 'npm run dev' in the frontend directory to access the UI.")
    print("\nPress Ctrl+C to terminate both processes.\n")

    try:
        while True:
            # Print a few lines from each to show they are alive
            line_api = api_proc.stdout.readline()
            if line_api:
                print(f"[API] {line_api.strip()}")
            
            line_sch = scheduler_proc.stdout.readline()
            if line_sch:
                print(f"[SCHEDULER] {line_sch.strip()}")
            
            if not line_api and not line_sch:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        api_proc.terminate()
        scheduler_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    launch()
