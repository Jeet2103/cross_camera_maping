# 📄 Brief Report: Cross-View Player Mapping & Re-identification System

The objective of this project is to build an advanced cross-view player tracking and re-identification system that effectively maps and tracks players across broadcast and tactical (top-down) camera views. In sports analytics, especially in football or basketball, player identities often become ambiguous when switching between different camera feeds. This system addresses that challenge by ensuring consistent player identification across disjoint views, enabling deeper insights into player movements, formations, and tactical decisions.

## 🔄 Detailed Workflow Summary

### 📌 1. Player Detection with YOLOv11

For precise and high-speed detection of players, the system utilizes the YOLOv11 (You Only Look Once version 11) object detection framework—renowned for its cutting-edge performance on both GPU and edge devices.

- **Model Used:** A fine-tuned YOLOv11 model named best.pt, trained on a sports-specific dataset to recognize multiple classes relevant to the game:
    - Player, Goalkeeper, Referee, and Ball.
- **Capabilities:**

    - **Multi-Class Detection:** Accurately detects and differentiates between players, goalkeepers, referees, and the ball in real-time.

    - **Color-Coded Bounding Boxes:** Each class is rendered with a distinct bounding box color for intuitive visual feedback.

    - **Advanced Detection Head:** Enhanced localization and classification, especially in high-density scenes with overlapping individuals.

    - **Temporal Consistency:** Reduces false positives and ID switches during rapid game transitions.

- **Output:**
The model outputs bounding boxes with class labels and confidence scores, which are passed directly to the tracking module for identity persistence.

🧾 `detect_players.py` — Player Detection Script
Below is the implementation of the detection module using the `best.pt` YOLOv11 model:

