# Cross Camera Mapping

**Cross Camera Mapping** is a computer vision system designed to accurately detect, track, and map players across multiple camera feeds—such as broadcast and tacticam views—using state-of-the-art object detection, feature extraction, and identity matching techniques.

---

## 📁 Project Structure


---

## 🔁 Workflow

1. **Input Videos**  
   - Place the broadcast and tacticam videos inside the `data/` folder.

2. **Player Detection**  
   - Run `detect_players.py` to detect players in both views using YOLOv5.  
   - Outputs stored in `detections/`.

3. **Player Tracking**  
   - Use `track_players.py` with StrongSORT to generate consistent player IDs across frames.  
   - Tracked results are saved in `tracking/`.

4. **Homography Computation**  
   - Run `compute_homography.py` to calculate spatial transformation (homography) between the two views.  
   - This enables mapping coordinates from one view to another.

5. **Appearance Feature Extraction**  
   - Execute `extract_features.py` to extract deep visual features using the OSNet re-ID model for each tracked player.  
   - Feature vectors are saved for matching.

6. **Cross-View Matching**  
   - Launch `match_players.py` to match players across views using FAISS + Hungarian algorithm.  
   - Combines appearance features and spatial position via homography for high-accuracy mapping.

7. **Visualization**  
   - Run `visualize_results.py` to render the matched players between the two views.  
   - Final annotated videos and overlays are stored in the `output/` directory.

8. **Pipeline Automation**  
   - Use `pipeline.py` to run the entire pipeline from detection to visualization in one go.

9. **Logging**  
   - All steps utilize a centralized logger defined in `logger_config/logger.py`  
   - Logs are stored in the `logs/` folder for debugging and analysis.

---

## ✅ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
