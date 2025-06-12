# Cross Camera Mapping Project

The objective of this project is to build an advanced cross-view player tracking and re-identification system that effectively maps and tracks players across broadcast and tactical (top-down) camera views. In sports analytics, especially in football or basketball, player identities often become ambiguous when switching between different camera feeds. This system addresses that challenge by ensuring consistent player identification across disjoint views, enabling deeper insights into player movements, formations, and tactical decisions.


## Key Features
- **Multi-Camera Tracking**: Seamless player identification across different views
- **Advanced Algorithms**: Combines YOLOv11, StrongSORT, and OSNet
- **Real-Time Processing**: Efficient pipeline for sports analytics
- **Precise Mapping**: Homography transformation for accurate position mapping
- **Fast Matching**: FAISS-based similarity search for player re-identification

## Technical Stack
- **Detection**: YOLOv11
- **Tracking**: StrongSORT
- **Features Extraction**: OSNet
- **Geometry**: Homography Transformation
- **Matching players**: FAISS

---
## Project Structure
```
cross_camera_mapping/
├── data/ # Input video files
│ ├── broadcast.mp4 # Broadcast camera feed
│ └── tacticam.mp4 # Tactical camera feed
│
├── model/ # Model weights
│ └── best.pt # YOLOv11 model file
│
├── detections/ # Player detection outputs(.csv file)
├── tracking/ # Player tracking data(.csv file)
├── mapping/ # Camera mapping data(.npy files and json file)
├── output/ # Final output files(.mp4 file)
├── logs/ # System logs
│
├── logger_config/ # Logging configuration
│ └── logger.py # Logger implementation
│
├── compute_homography.py # Homography calculations
├── pipeline.py # Main processing pipeline
│
├── detect_players.py # Player detection script
├── track_players.py # Player tracking script
├── extract_features.py # Feature extraction
├── match_players.py # Player matching across views
├── visualize_results.py # Visualization utilities
│
└── requirements.txt # Project dependencies
```
---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Jeet2103/cross_camera_maping.git
cd cross_camera_maping

```


### 2. Create and Activate a Virtual Environment

You can use either `venv` or `conda` to manage your environment.

**Using `venv`:**

- **For Windows:**
  - Create: `python -m venv venv`
  - Activate: `venv\Scripts\activate`

- **For Linux/MacOS:**
  - Create: `python3 -m venv venv`
  - Activate: `source venv/bin/activate`

**Using `conda`:**

- Create: `conda create -n ccm_env python=3.9`
- Activate: `conda activate ccm_env`

### 3. Install Dependencies

Install all required dependencies using the following command:


`pip install -r requirements.txt`

## Dependencies

The project requires the following Python packages:

- ultralytics  
- opencv-python  
- torch  
- torchvision  
- numpy  
- pandas  
- matplotlib  
- deep-sort-realtime  
- scikit-learn  
- torchreid  
- tensorboard  
- cython  
- charset-normalizer  
- chardet  
- faiss-cpu  

Python version: **3.9 or above**


### 4: Download YOLOv11 Model

Download the YOLOv11 model from the following link:

[Download YOLOv11](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)

After downloading, place the model file (`best.pt`) inside the `model/` directory:



## Running the Project

After installation, run the main pipeline using:

`python pipeline.py`

This will perform detection, tracking, feature extraction, and cross-camera matching, and generate output with visual identity mapping.

## Contact

Maintained by **Jeet Nandigrami**  
- GitHub: [Jeet2103](https://github.com/Jeet2103)  
- LinkedIn: [Jeet Nandigrami](https://www.linkedin.com/in/jeet-nandigrami/)
- Resume : [RESUME](https://drive.google.com/file/d/1Zvm0yAK--t_K-lNBpLnDFA2Lz41ZBqvX/view?usp=sharing)
- Email : jeetnandigrami2103@gmail.com


# **Report Documentation File** :Report.md
