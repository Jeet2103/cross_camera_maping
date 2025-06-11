# Project Structure: cross_camera_mapping/

# ├── data/
# │   ├── broadcast.mp4
# │   └── tacticam.mp4
# ├── model/
# │   └── best.pt
# ├── detections/
# ├── tracking/
# ├── mapping/
# ├── output/
# ├── logs/
# ├── logger_config/
    #   ├── logger.py
# ├── compute_homography.py/
# ├── pipeline.py/
# ├── detect_players.py
# ├── track_players.py
# ├── extract_features.py
# ├── match_players.py
# └── visualize_results.py
# └── requirements.txt


import os
from logger_config.logger import get_logger

# Initialize logger
logger = get_logger('setup.log')

def create_project_structure():
    """
    Create the necessary directory and file structure for the cross_camera_mapping project.
    Logs all operations instead of printing them.
    """
    # === List of folders to be created ===
    directories = [
        'data',
        'model',
        'detections',
        'tracking',
        'mapping',
        'output',
        'counts',
        'logs',
        'logger_config'
    ]
    
    # === List of core Python and config files to be created ===
    files = [
        'detect_players.py',
        'track_players.py',
        'extract_features.py',
        'match_players.py',
        'visualize_results.py',
        'requirements.txt',
        'compute_homography.py',
        'pipeline.py',
        'logger_config/logger.py',
        'README.md'
    ]
    
    # === Create directories ===
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory '{directory}': {e}")
    
    # === Create empty Python and config files ===
    for py_file in files:
        try:
            with open(py_file, 'w') as f:
                # Add shebang and basic header
                f.write('#!/usr/bin/env python\n\n')
            logger.info(f"Created file: {py_file}")
        except Exception as e:
            logger.error(f"Failed to create file '{py_file}': {e}")

# === Entry Point ===
if __name__ == '__main__':
    logger.info("Initializing project structure setup.")
    create_project_structure()
    logger.info("Project structure setup complete.")
