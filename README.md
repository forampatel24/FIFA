# AI Tactical Analyst

Automated football tactical analysis using computer vision. Detects players, tracks movement, classifies teams, maps pitch positions, and analyzes formations — all from a match video.

> **Status:** Work in progress. Core pipeline works end-to-end but accuracy and UX are actively being improved.

## Features

- **Player Detection** — YOLOv8 detects players, goalkeepers, referees, and ball
- **Multi-Object Tracking** — ByteTrack assigns persistent IDs across frames
- **Team Classification** — K-Means clustering on jersey colors to separate teams
- **Pitch Mapping** — Homography transforms pixel coordinates to real-world pitch positions
- **Formation Detection** — Automatic 4-4-2, 4-3-3, 3-5-2 detection via row clustering
- **Heatmaps** — Player and team movement density visualization
- **Match Timeline** — Formation changes over time
- **Player Metrics** — Distance covered, speed, average position
- **Tactical Dashboard** — Streamlit UI with 5 analysis tabs

## Project Structure

```
FIFA/
├── app/                  # Streamlit dashboard
│   └── streamlit_app.py
├── detection/            # YOLOv8 detector
│   ├── detector.py       # Detection wrapper + class mapping
│   └── model.py          # Model loader
├── tracking/             # ByteTrack wrapper
│   └── tracker.py
├── homography/           # Pitch calibration
│   └── homography.py
├── analytics/            # Analysis modules
│   ├── formations.py     # Formation analyzer (K-Means row clustering)
│   ├── heatmaps.py       # Heatmap generator
│   └── metrics.py        # Player/team metrics calculator
├── visualization/        # Plotting
│   ├── pitch.py          # mplsoccer pitch visualizer
│   └── timeline.py       # Formation timeline renderer
├── utils/                # Utilities
│   ├── team_classifier.py # K-Means jersey color classifier
│   └── video_utils.py    # Video loader + metadata
├── pipeline.py           # Main processing orchestrator
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- See `requirements.txt` for full list

Key dependencies: ultralytics, supervision, opencv-python, numpy, scikit-learn, streamlit, plotly, mplsoccer, matplotlib, scipy

## Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8x model (auto-downloaded on first run, or place yolov8x.pt in project root)
```

## Usage

```powershell
streamlit run app/streamlit_app.py
```

1. Upload a match video (MP4, AVI, MOV, MKV — 720p or lower recommended)
2. Configure detection settings
3. Click **Start Analysis**
4. Explore results across 5 tabs

### Settings

| Setting | Description | Recommended |
|---|---|---|
| Detection Confidence | Minimum confidence for detections | 0.3–0.5 |
| Processing Quality | Process every Nth frame | 10–30 (higher = faster) |
| Model Size | YOLOv8 variant (n/s/m/l/x) | s or m on CPU, x on GPU |

## Performance Notes

- **CPU-only**: YOLOv8x processes ~1 frame every 5-10 seconds on CPU. Use `s` or `m` model sizes for faster processing.
- **GPU**: With CUDA, YOLOv8x processes 30+ FPS.
- **Video resolution**: Processing 4K video is very slow on CPU. Downscale to 720p or use the smaller model.
- **Tracking stability**: Processing fewer frames (higher `process_every_n`) may cause tracking ID fragmentation. The tracker now filters short-lived IDs from results.

## Current Limitations

- Team classification works best on videos with clearly distinct jersey colors
- Homography calibration assumes a standard green football pitch
- Formation detection requires sufficient players per team (minimum 3 rows)
- Ball detection and tracking is basic (no trajectory smoothing or possession analysis)
- No team possession or pass analysis yet
- Processing time scales with video length and resolution

## Roadmap

- [ ] Improve tracking consistency across frame skips
- [ ] Add trajectory smoothing and interpolation
- [ ] Implement possession analysis
- [ ] Add pass network visualization
- [ ] Support multiple camera angles
- [ ] Real-time processing with GPU support
- [ ] Export match reports (PDF)

## License

MIT
