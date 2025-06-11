import os
import argparse
import subprocess
from logger_config.logger import get_logger

# Initialize logger
logger = get_logger("pipeline.log")

def run_stage(stage_name, command):
    """
    Runs a specific stage of the pipeline using subprocess.
    Logs the status of each stage and exits if any stage fails.
    """
    logger.info(f"Starting stage: {stage_name}")
    print(f"\nRunning stage: {stage_name}")
    result = subprocess.run(command, shell=True)
    
    if result.returncode == 0:
        logger.info(f"{stage_name} completed successfully.")
        print(f"{stage_name} completed.")
    else:
        logger.error(f"{stage_name} failed. Command: {command}")
        print(f"{stage_name} failed. Exiting.")
        exit(1)

def main():
    """
    Main function to run the entire cross-view player mapping pipeline.
    It ensures all folders exist and executes each stage in order.
    """
    parser = argparse.ArgumentParser(description="Cross-view Player Mapping Pipeline")
    parser.add_argument('--broadcast', type=str, default='data/broadcast.mp4', help="Path to broadcast video")
    parser.add_argument('--tacticam', type=str, default='data/tacticam.mp4', help="Path to tacticam video")
    args = parser.parse_args()

    # Ensure all required folders exist
    required_dirs = ['data', 'mapping', 'tracking', 'detections', 'output', 'counts']
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Ensured directory exists: {d}")

    # Log input video paths
    logger.info(f"Using broadcast video: {args.broadcast}")
    logger.info(f"Using tacticam video: {args.tacticam}")

    # Run each pipeline stage sequentially
    run_stage("Compute Homography", "python compute_homography.py")
    run_stage("Detect Players", "python detect_players.py")
    run_stage("Track Players", "python track_players.py")
    run_stage("Extract Features", "python extract_features.py")
    run_stage("Match Players", "python match_players.py")
    run_stage("Visualize Results", "python visualize_results.py")

    logger.info("All pipeline stages executed successfully.")
    print("\nAll pipeline stages completed! Check the 'output/' folder for the final visualizations.")

if __name__ == "__main__":
    main()
