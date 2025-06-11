# Cross Camera Mapping

Cross Camera Mapping is a computer vision pipeline designed to track and match players across two different camera feeds (e.g., broadcast view and tacticam) using deep learning-based detection, tracking, appearance feature extraction, and spatial matching.

## Project Structure


## Setup Instructions

### 1. Clone the Repository

Clone the project from GitHub:

`git clone https://github.com/Jeet2103/cross_camera_maping.git`
`cd cross_camera_maping`


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

## Running the Project

After installation, run the main pipeline using:

`python pipeline.py`

This will perform detection, tracking, feature extraction, and cross-camera matching, and generate output with visual identity mapping.

## Contact

Maintained by **Jeet Nandigrami**  
- GitHub: [Jeet2103](https://github.com/Jeet2103)  
- LinkedIn: [Jeet Nandigrami](https://www.linkedin.com/in/jeet-nandigrami/)
- Resume : [RESUME](https://drive.google.com/file/d/1Zvm0yAK--t_K-lNBpLnDFA2Lz41ZBqvX/view?usp=sharing)