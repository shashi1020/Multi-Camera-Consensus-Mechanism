# SENTINEL NEXUS: Advanced Multi-Camera Adaptive Threat Detection

> **Research-only prototype**: a safety wrapper around YOLO-based object detection that:
> - Fuses detections from multiple cameras,
> - Estimates uncertainty and cross-camera disagreement,
> - Detects suspicious inputs,
> - Mitigates them via robust models or test-time augmentation (TTA),
> - Logs everything for later analysis.

This repository provides an **adaptive multi-camera consensus framework** with a built-in **adversarial defense pipeline** for vision models. It is designed to sit *around* standard YOLO models (from [Ultralytics](https://github.com/ultralytics/ultralytics)) and make them:

- More robust across **multiple cameras or viewpoints**,
- More aware of **uncertainty and disagreement**,
- More resistant to **input quality issues or potential adversarial attacks**.

---

## Key Ideas & Innovation

- ✅ **Multi-camera consensus**: Normalize detections per camera and cluster them across views to create a shared “object hypothesis”.
- ✅ **Adaptive camera reliability**: Each camera gets a dynamic reliability score, which is updated as it agrees/disagrees with multi-camera consensus.
- ✅ **Spatio-temporal validation**: Objects must persist across frames and cameras (spatially consistent) before being fully trusted.
- ✅ **Threat scoring**: Combines:
  - Consensus **uncertainty**,
  - Cross-camera **disagreement**,
  - **Image quality anomalies** (blur, extreme brightness),
  into a single `threat_score ∈ [0, 1]`.
- ✅ **Defense pipeline**:
  - If `threat_score` is high:
    - Log all frames + metadata,
    - Optionally rerun detection with a **robust model** (if latency budget allows),
    - Otherwise, use **test-time augmentation (TTA)** with the fast model.
- ✅ **Suspicious sample logging**:
  - Per-camera frames are saved with hashed filenames,
  - A structured JSON log captures threat scores, meta-data, and consensus details,
  - Perfect for building a red-team or adversarial dataset.
---

## Repository Structure

A minimal layout for this project:

```text
.
├── adaptive_multicam.py      # Main script (the file in this README)
├── utils/
│   └── coco.txt              # Class list (one class name per line)
├── models/
│   ├── fast_model.pt         # Your fast YOLO weights (example)
│   └── robust_model.pt       # Optional robust YOLO weights
├── threat_logs/              # Auto-created directory for suspicious samples
└── README.md

![Multi-Camera_Consensus_Algorithm_-_Intelligent_Fusion_ _Weighted_Voting_System](https://github.com/user-attachments/assets/a75fedff-bfa4-43e4-bd95-ac9b53837cee)


