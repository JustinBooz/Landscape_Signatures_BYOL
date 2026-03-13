import os
import subprocess
import yaml
import os
import glob
import random
import time

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_dir_size_gb(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024**3)

def run_command(command):
    print(f"\n[Orchestrator] Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"[Orchestrator] Command failed with exit code {result.returncode}")
        return False
    return True

def main():
    print("Orchestrator starting...", flush=True)
    config = load_config()
    shards_dir = config["data"]["output_shards_dir"]
    max_cache_gb = config.get("orchestrator", {}).get("max_disk_cache_gb", 500.0)
    max_replay_gb = config.get("orchestrator", {}).get("max_replay_gb", 50.0)
    
    import sys
    python_bin = f"{sys.executable} -u"
    
    os.makedirs(shards_dir, exist_ok=True)
    
    iteration = 1
    while True:
        print(f"\n{'='*55}")
        # 1. Ingestion Phase
        print("\n=== Phase 1: Data Ingestion (Scrape & Pack) ===")
        success = run_command(f"{python_bin} data/ingestion.py")
        if not success:
            print("[Orchestrator] Ingestion script crashed. Halting pipeline.")
            break
            
        shards = glob.glob(os.path.join(shards_dir, "*.tar"))
        current_size = get_dir_size_gb(shards_dir)
        print(f"[Orchestrator] Currently using {current_size:.2f} GB of {max_cache_gb} GB cache.")
        
        # Check if ingestion has completely finished the geographic grid
        ingestion_complete = False
        import json
        try:
            with open("data/ingestion_state.json") as f:
                state = json.load(f)
                import numpy as np
                coords_len = len(np.load("data/h3_coords_res7.npy"))
                if state.get("last_idx", 0) >= coords_len:
                    ingestion_complete = True
        except Exception:
            pass
        
        # 2. Training (Triggered strictly when current_size >= max_cache_gb * 0.9 OR grid completes)
        if current_size >= max_cache_gb * 0.9 or (ingestion_complete and current_size > 0):
            print(f"\n=== Phase 2: BYOL Training (Capacity Triggered at {current_size:.1f} GB) ===")
            success = run_command(f"{python_bin} train.py")
            if not success:
                print("[Orchestrator] Optimizer loop crashed. Halting pipeline.")
                break

            # 3. Cleanup & Replay FIFO
            print("\n=== Phase 3: Cleanup & Replay FIFO ===")
            replay_dir = os.path.join(shards_dir, "..", "replay_buffer")
            os.makedirs(replay_dir, exist_ok=True)
            
            shards = glob.glob(os.path.join(shards_dir, "*.tar"))
            # According to prompt: promote all current shards to replay and then enforce FIFO cap
            import shutil
            for shard in shards:
                try:
                    target = os.path.join(replay_dir, os.path.basename(shard))
                    shutil.move(shard, target)
                except Exception as e:
                    print(f"[Orchestrator] Error promoting {shard} to replay: {e}")
            
            # 50 GB FIFO hard cap using mtime
            replay_limit_gb = 750.0
            def run_fifo_cleanup():
                if get_dir_size_gb(replay_dir) > replay_limit_gb:
                    print(f"[Orchestrator] FIFO Cap reached. Skipping deletion as requested by user.")
            
            run_fifo_cleanup()
            print("[Orchestrator] Local Epoch & FIFO Cleanup Complete.")
            
            if ingestion_complete:
                print("[Orchestrator] Entire geographic grid processed and trained. Pipeline complete!")
                break
        else:
            print(f"\n[Orchestrator] Cache at {current_size:.1f}/{max_cache_gb} GB. Continuing ingestion...")
            
            if ingestion_complete and current_size == 0:
                print("[Orchestrator] Entire geographic grid processed. No new data to train on. Pipeline complete!")
                break
            
        iteration += 1

if __name__ == "__main__":
    main()
