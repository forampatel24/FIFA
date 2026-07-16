import os
import sys
import json
import tempfile
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from pipeline import ProcessingPipeline
from visualization.pitch import PitchVisualizer
from visualization.timeline import TimelineVisualizer
from utils.video_utils import VideoLoader, get_file_size_str
from analytics.heatmaps import HeatmapGenerator

st.set_page_config(
    page_title="FIFA Tactical Analyst",
    page_icon="\u26bd",
    layout="wide",
    initial_sidebar_state="expanded",
)

FIFA_BLACK = "#0D0D0D"
FIFA_WHITE = "#FFFFFF"
FIFA_RED = "#E41A1C"
FIFA_BLUE = "#1A237E"
FIFA_GOLD = "#FFD700"
FIFA_GRAY = "#1E1E1E"
FIFA_LIGHT_GRAY = "#2A2A2A"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#AAAAAA"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {{ font-family: 'Inter', sans-serif; }}
    
    .stApp {{
        background: {FIFA_BLACK};
        color: {TEXT_PRIMARY};
    }}
    
    div[data-testid="stSidebar"] {{
        background: {FIFA_GRAY};
        border-right: 1px solid {FIFA_LIGHT_GRAY};
    }}
    
    .fifa-header {{
        background: linear-gradient(135deg, {FIFA_BLACK} 0%, {FIFA_BLUE} 100%);
        padding: 20px 30px;
        border-radius: 12px;
        border-left: 4px solid {FIFA_RED};
        margin-bottom: 20px;
    }}
    
    .fifa-title {{
        color: {FIFA_WHITE};
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
    }}
    
    .fifa-subtitle {{
        color: {FIFA_GOLD};
        font-size: 0.9rem;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 0;
    }}
    
    .fifa-badge {{
        display: inline-block;
        background: linear-gradient(135deg, {FIFA_RED}, #FF4444);
        color: white;
        font-weight: 700;
        font-size: 0.65rem;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    
    .card {{
        background: {FIFA_GRAY};
        border: 1px solid {FIFA_LIGHT_GRAY};
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        transition: border-color 0.2s;
    }}
    .card:hover {{
        border-color: {FIFA_RED};
    }}
    
    .card-label {{
        color: {TEXT_SECONDARY};
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }}
    
    .card-value {{
        color: {FIFA_WHITE};
        font-size: 1.8rem;
        font-weight: 700;
    }}
    
    .card-value-small {{
        color: {FIFA_WHITE};
        font-size: 1.1rem;
        font-weight: 600;
    }}
    
    .stButton button {{
        background: linear-gradient(135deg, {FIFA_RED}, #FF4444);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        transition: transform 0.15s, box-shadow 0.15s;
        box-shadow: 0 4px 15px rgba(228, 26, 28, 0.3);
    }}
    .stButton button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(228, 26, 28, 0.5);
    }}
    .stButton button:active {{
        transform: translateY(0);
    }}
    
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {FIFA_RED}, #FF4444);
    }}
    
    div[data-testid="stTabs"] button {{
        color: {TEXT_SECONDARY};
        font-weight: 600;
        font-size: 0.85rem;
    }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {FIFA_RED};
        border-bottom-color: {FIFA_RED};
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {FIFA_WHITE};
        font-weight: 700;
    }}
    
    p, li, span, div {{
        color: {TEXT_SECONDARY};
    }}
    
    .stDataFrame {{
        border-radius: 8px;
        overflow: hidden;
    }}
    .stDataFrame thead tr th {{
        background: {FIFA_LIGHT_GRAY};
        color: {TEXT_PRIMARY};
    }}
    
    .sidebar-section {{
        color: {FIFA_GOLD};
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 20px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 1px solid {FIFA_LIGHT_GRAY};
    }}
    
    .stSelectbox label, .stSlider label, .stRadio label {{
        color: {TEXT_SECONDARY};
        font-weight: 500;
    }}
    
    .status-box {{
        background: {FIFA_LIGHT_GRAY};
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }}
    
    footer {{ display: none; }}
</style>
""", unsafe_allow_html=True)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "results" not in st.session_state:
    st.session_state.results = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "processing" not in st.session_state:
    st.session_state.processing = False
if "progress" not in st.session_state:
    st.session_state.progress = 0.0
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "annotated_video_path" not in st.session_state:
    st.session_state.annotated_video_path = None
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None
if "selected_player" not in st.session_state:
    st.session_state.selected_player = None
if "selected_team" not in st.session_state:
    st.session_state.selected_team = 0
if "timeline_frame" not in st.session_state:
    st.session_state.timeline_frame = 0
if "processing_error" not in st.session_state:
    st.session_state.processing_error = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = ""


def handle_tactical_pitch(results, pipeline):
    col1, col2 = st.columns([3, 1])

    with col2:
        max_frames = max(results["video_metadata"].get("processed_frames", 100), 1)
        timeline_frame = st.slider(
            "Match Timeline",
            min_value=0,
            max_value=max_frames,
            value=min(st.session_state.timeline_frame, max_frames),
            step=max(1, max_frames // 100),
        )
        st.session_state.timeline_frame = timeline_frame

        show_ball = st.checkbox("Show Ball", value=True)
        show_trails = st.checkbox("Movement Trails", value=False)

        st.markdown("### Current Formation")
        formation_history = results.get("formation_history", [])
        if formation_history:
            closest = min(formation_history, key=lambda x: abs(x["frame"] - timeline_frame))
            st.markdown(
                f'<div class="card">'
                f'<div class="card-label">Team A</div>'
                f'<div class="card-value-small">{closest["team_a_formation"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="card">'
                f'<div class="card-label">Team B</div>'
                f'<div class="card-value-small">{closest["team_b_formation"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col1:
        p0, p1, ball, unassigned = pipeline.get_frame_positions(timeline_frame)

        viz = PitchVisualizer()

        if show_trails:
            team_a_trails = {}
            team_b_trails = {}
            unassigned_trails = {}
            for pid, stats in results.get("player_stats", {}).items():
                trail = stats.get("positions", [])
                if stats["team"] == 0 and len(trail) > 1:
                    team_a_trails[pid] = trail
                elif stats["team"] == 1 and len(trail) > 1:
                    team_b_trails[pid] = trail
                elif len(trail) > 1:
                    unassigned_trails[pid] = trail

            ball_trail = []
            for f_idx, pos in results.get("ball_trajectory", []):
                if f_idx <= timeline_frame:
                    ball_trail.append(pos)

            buf = viz.plot_movement_trails(
                team_a_trails, team_b_trails,
                ball_trail if show_ball else None,
                unassigned_trails if show_ball else None,
            )
        else:
            ball_pos = ball if show_ball else None
            buf = viz.plot_positions(p0, p1, unassigned, ball_pos)

        st.image(buf, use_container_width=True)

        player_stats = results.get("player_stats", {})
        if player_stats:
            positions_df = pd.DataFrame([
                {"Player ID": pid, "Team": f"Team {s['team']}",
                 "X (m)": round(s["avg_position"]["x"], 1),
                 "Y (m)": round(s["avg_position"]["y"], 1),
                 "Distance (m)": round(s["metrics"]["distance_covered"], 1),
                 "Avg Speed (m/s)": round(s["metrics"]["avg_speed"], 1)}
                for pid, s in player_stats.items()
            ])
            st.dataframe(positions_df, use_container_width=True, hide_index=True)


def handle_team_analysis(results):
    player_stats = results.get("player_stats", {})
    team_a_stats = {pid: s for pid, s in player_stats.items() if s["team"] == 0}
    team_b_stats = {pid: s for pid, s in player_stats.items() if s["team"] == 1}
    ta = results.get("team_a_metrics", {})
    tb = results.get("team_b_metrics", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card"><div class="card-label">Formation</div>'
                    f'<div class="card-value-small">{results.get("dominant_formation_team_a", "?")}</div></div>',
                    unsafe_allow_html=True)
        metrics_a = pd.DataFrame({
            "Metric": ["Width", "Depth", "Compactness", "Defensive Line", "Attacking Line", "Players"],
            "Value": [
                f"{ta.get('width', 0):.1f}m",
                f"{ta.get('depth', 0):.1f}m",
                f"{ta.get('compactness', 0):.1f}m",
                f"{ta.get('defensive_line', 0):.1f}m",
                f"{ta.get('attacking_line', 0):.1f}m",
                ta.get("num_players", 0),
            ],
        })
        st.dataframe(metrics_a, use_container_width=True, hide_index=True)

        if team_a_stats:
            st.markdown("#### Avg Position Map")
            avg_pos = {pid: s["avg_position"] for pid, s in team_a_stats.items()}
            buf = PitchVisualizer().plot_average_positions(avg_pos, {})
            st.image(buf, use_container_width=True)

            st.markdown("#### Team Heatmap")
            all_x = np.array([s["avg_position"]["x"] for s in team_a_stats.values()])
            all_y = np.array([s["avg_position"]["y"] for s in team_a_stats.values()])
            if len(all_x) >= 3:
                positions = np.column_stack([all_x, all_y])
                buf = PitchVisualizer().plot_heatmap(positions)
                st.image(buf, use_container_width=True)

    with col2:
        st.markdown('<div class="card"><div class="card-label">Formation</div>'
                    f'<div class="card-value-small">{results.get("dominant_formation_team_b", "?")}</div></div>',
                    unsafe_allow_html=True)
        metrics_b = pd.DataFrame({
            "Metric": ["Width", "Depth", "Compactness", "Defensive Line", "Attacking Line", "Players"],
            "Value": [
                f"{tb.get('width', 0):.1f}m",
                f"{tb.get('depth', 0):.1f}m",
                f"{tb.get('compactness', 0):.1f}m",
                f"{tb.get('defensive_line', 0):.1f}m",
                f"{tb.get('attacking_line', 0):.1f}m",
                tb.get("num_players", 0),
            ],
        })
        st.dataframe(metrics_b, use_container_width=True, hide_index=True)

        if team_b_stats:
            st.markdown("#### Avg Position Map")
            avg_pos = {pid: s["avg_position"] for pid, s in team_b_stats.items()}
            buf = PitchVisualizer().plot_average_positions({}, avg_pos)
            st.image(buf, use_container_width=True)

            st.markdown("#### Team Heatmap")
            all_x = np.array([s["avg_position"]["x"] for s in team_b_stats.values()])
            all_y = np.array([s["avg_position"]["y"] for s in team_b_stats.values()])
            if len(all_x) >= 3:
                positions = np.column_stack([all_x, all_y])
                buf = PitchVisualizer().plot_heatmap(positions)
                st.image(buf, use_container_width=True)

    st.markdown("### Comparison Chart")
    if team_a_stats and team_b_stats:
        comp_data = pd.DataFrame({
            "Player": [f"#{pid}" for pid in list(team_a_stats.keys()) + list(team_b_stats.keys())],
            "Team": ["Team A"] * len(team_a_stats) + ["Team B"] * len(team_b_stats),
            "Distance (m)": [s["metrics"]["distance_covered"] for s in team_a_stats.values()] +
                            [s["metrics"]["distance_covered"] for s in team_b_stats.values()],
            "Avg Speed (m/s)": [s["metrics"]["avg_speed"] for s in team_a_stats.values()] +
                               [s["metrics"]["avg_speed"] for s in team_b_stats.values()],
        })

        fig = px.bar(
            comp_data, x="Player", y="Distance (m)",
            color="Team", barmode="group",
            title="Distance Covered by Player",
            color_discrete_map={"Team A": "#e41a1c", "Team B": "#377eb8"},
        )
        fig.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0D0D0D",
            font_color="#AAAAAA", title_font_color="#FFFFFF",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            comp_data, x="Avg Speed (m/s)", y="Distance (m)",
            color="Team", size="Distance (m)",
            hover_data=["Player"],
            title="Speed vs Distance",
            color_discrete_map={"Team A": "#e41a1c", "Team B": "#377eb8"},
        )
        fig2.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0D0D0D",
            font_color="#AAAAAA", title_font_color="#FFFFFF",
        )
        st.plotly_chart(fig2, use_container_width=True)


def handle_player_analysis(results):
    player_stats = results.get("player_stats", {})
    if not player_stats:
        st.info("No player data available.")
        return

    team_choice = st.radio("Select Team", ["Team A", "Team B"], horizontal=True)
    team_id = 0 if team_choice == "Team A" else 1
    team_players = {pid: s for pid, s in player_stats.items() if s["team"] == team_id}

    if not team_players:
        st.info(f"No players found for {team_choice}.")
        return

    player_ids = list(team_players.keys())
    selected_id = st.selectbox(
        "Select Player", options=player_ids,
        format_func=lambda x: f"Player #{x}",
    )

    if selected_id in team_players:
        stats = team_players[selected_id]
        metrics = stats["metrics"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<div class="card"><div class="card-label">Player</div>'
                f'<div class="card-value-small">#{selected_id}</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="card"><div class="card-label">Team</div>'
                f'<div class="card-value-small">{team_choice}</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="card"><div class="card-label">Avg Position</div>'
                f'<div class="card-value-small">({stats["avg_position"]["x"]:.1f}, {stats["avg_position"]["y"]:.1f})</div></div>',
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f'<div class="card"><div class="card-label">Frames</div>'
                f'<div class="card-value-small">{metrics["frames_tracked"]}</div></div>',
                unsafe_allow_html=True,
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="card"><div class="card-label">Distance</div>'
                f'<div class="card-value">{metrics["distance_covered"]:.1f}m</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="card"><div class="card-label">Avg Speed</div>'
                f'<div class="card-value">{metrics["avg_speed"]:.2f} m/s</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="card"><div class="card-label">Max Speed</div>'
                f'<div class="card-value">{metrics["max_speed"]:.2f} m/s</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        col1, col2 = st.columns(2)

        positions = stats.get("positions", [])

        with col1:
            st.markdown("#### Player Heatmap")
            if len(positions) >= 3:
                arr = np.array(positions)
                buf = PitchVisualizer().plot_heatmap(arr)
                st.image(buf, use_container_width=True)

        with col2:
            st.markdown("#### Movement Trail")
            if positions:
                team_trails = {selected_id: positions}
                buf = PitchVisualizer().plot_movement_trails(
                    team_trails if team_id == 0 else {},
                    team_trails if team_id == 1 else {},
                )
                st.image(buf, use_container_width=True)

        st.markdown("#### Position Timeline")
        if positions:
            pos_df = pd.DataFrame(positions, columns=["X", "Y"])
            pos_df["Frame"] = range(len(positions))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pos_df["Frame"], y=pos_df["X"],
                mode="lines", name="X Position",
                line=dict(color="#e41a1c"),
            ))
            fig.add_trace(go.Scatter(
                x=pos_df["Frame"], y=pos_df["Y"],
                mode="lines", name="Y Position",
                line=dict(color="#377eb8"),
            ))
            fig.update_layout(
                title="Player Position Over Time",
                xaxis_title="Frame", yaxis_title="Position (m)",
                plot_bgcolor="#1a1a2e", paper_bgcolor="#0D0D0D",
                font_color="#AAAAAA", title_font_color="#FFFFFF",
            )
            st.plotly_chart(fig, use_container_width=True)


def handle_timeline(results, pipeline):
    formation_history = results.get("formation_history", [])
    total_frames = results["video_metadata"].get("processed_frames", 0)
    fps = results["video_metadata"].get("fps", 30)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Formation Timeline")
        if formation_history:
            buf = TimelineVisualizer().render(
                formation_history,
                total_frames=total_frames, fps=fps,
            )
            st.image(buf, use_container_width=True)

            st.markdown("### Formation Changes")
            changes = results.get("formation_changes", [])
            if changes:
                changes_data = []
                for c in changes:
                    t = c["frame"] / fps
                    mins = int(t // 60)
                    secs = int(t % 60)
                    changes_data.append({
                        "Time": f"{mins:02d}:{secs:02d}",
                        "Team A Formation": c["team_a_formation"],
                        "Team B Formation": c["team_b_formation"],
                    })
                st.dataframe(
                    pd.DataFrame(changes_data),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No formation changes detected.")
        else:
            st.info("No formation data available.")

    with col2:
        st.markdown("### Formation Distribution")
        history = results.get("formation_history", [])
        if history:
            formations_a = [h["team_a_formation"] for h in history]
            counter_a = Counter(formations_a)
            df_a = pd.DataFrame(counter_a.most_common(), columns=["Formation", "Frames"])

            st.markdown("**Team A**")
            st.dataframe(df_a, use_container_width=True, hide_index=True)

            fig_a = px.pie(
                values=df_a["Frames"], names=df_a["Formation"],
                title="Team A Formation Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_a.update_layout(
                plot_bgcolor="#1a1a2e", paper_bgcolor="#0D0D0D",
                font_color="#AAAAAA", title_font_color="#FFFFFF",
            )
            st.plotly_chart(fig_a, use_container_width=True)

    st.markdown("---")
    st.markdown("### Ball Trajectory")
    ball_traj = results.get("ball_trajectory", [])
    if ball_traj:
        ball_data = [{"Frame": f, "X": p["x"], "Y": p["y"]} for f, p in ball_traj]
        ball_df = pd.DataFrame(ball_data)

        fig = px.scatter(
            ball_df, x="X", y="Y",
            title="Ball Movement on Pitch",
            color_continuous_scale="Viridis",
            color="Frame",
            size=[2] * len(ball_df),
        )
        fig.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0D0D0D",
            font_color="#AAAAAA", title_font_color="#FFFFFF",
            xaxis=dict(range=[0, 105]),
            yaxis=dict(range=[0, 68]),
        )
        fig.update_traces(marker=dict(opacity=0.6))
        st.plotly_chart(fig, use_container_width=True)


st.markdown(
    f'<div class="fifa-header">'
    f'<div style="display:flex; align-items:center; gap:16px;">'
    f'<div style="font-size:2.5rem;">\u26bd</div>'
    f'<div>'
    f'<p class="fifa-subtitle">\u26a1 AI-Powered Analysis</p>'
    f'<h1 class="fifa-title">AI Tactical Analyst</h1>'
    f'</div>'
    f'<div style="margin-left:auto;">'
    f'<span class="fifa-badge">Powered by Computer Vision</span>'
    f'</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<p class="sidebar-section">Match Input</p>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload Match Video",
    type=["mp4", "avi", "mov", "mkv"],
    help="Supported: MP4, AVI, MOV, MKV",
)

if uploaded_file is not None:
    st.session_state.uploaded_filename = uploaded_file.name

st.sidebar.markdown('<p class="sidebar-section">Detection Settings</p>', unsafe_allow_html=True)

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.1, max_value=0.9, value=0.3, step=0.05,
)

process_every = st.sidebar.selectbox(
    "Processing Quality",
    options=[1, 2, 5, 10, 15],
    index=2,
    format_func=lambda x: f"Every {x} frame{'s' if x > 1 else ''}",
)

model_size = st.sidebar.selectbox(
    "Model Size",
    options=["n", "s", "m", "l", "x"],
    index=4,
    format_func=lambda x: {"n": "Nano", "s": "Small", "m": "Medium", "l": "Large", "x": "X-Large"}[x],
)

st.sidebar.markdown('<p class="sidebar-section">Actions</p>', unsafe_allow_html=True)

if uploaded_file is not None and not st.session_state.processing:
    if st.sidebar.button("Start Analysis", type="primary", use_container_width=True):
        temp_dir = Path(tempfile.mkdtemp())
        st.session_state.temp_dir = temp_dir
        video_path = temp_dir / "input_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.video_path = str(video_path)
        st.session_state.processing = True
        st.session_state.processed = False
        st.session_state.progress = 0.0
        st.session_state.processing_error = None
        st.rerun()

if st.session_state.processing and not st.session_state.processed:
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">'
        f'<span style="color:#FFD700;">\u26a0\ufe0f</span>'
        f'<span style="color:white; font-weight:600;">Processing...</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    model_path = str(ROOT_DIR / f"yolov8{model_size}.pt")
    output_dir = st.session_state.temp_dir / "analysis"

    pipeline = ProcessingPipeline(
        model_path=model_path,
        detection_confidence=confidence,
        process_every_n=process_every,
    )

    status_text.markdown(
        f'<div class="status-box">'
        f'<div style="display:flex; justify-content:space-between;">'
        f'<span style="color:#AAAAAA;">Status</span>'
        f'<span style="color:#FFD700; font-weight:600;">Processing video...</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    try:
        results = pipeline.process_video(
            st.session_state.video_path,
            output_dir=str(output_dir),
            progress_callback=lambda pct, frm, tot: (
                progress_bar.progress(pct / 100),
                status_text.markdown(
                    f'<div class="status-box">'
                    f'<div style="display:flex; justify-content:space-between; margin-bottom:4px;">'
                    f'<span style="color:#AAAAAA;">Status</span>'
                    f'<span style="color:#FFD700; font-weight:600;">Processing...</span>'
                    f'</div>'
                    f'<div style="display:flex; justify-content:space-between;">'
                    f'<span style="color:#AAAAAA;">Frames</span>'
                    f'<span style="color:white; font-weight:600;">{frm}/{tot}</span>'
                    f'</div>'
                    f'<div style="display:flex; justify-content:space-between;">'
                    f'<span style="color:#AAAAAA;">Progress</span>'
                    f'<span style="color:white; font-weight:600;">{pct:.1f}%</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                ),
            ),
        )
        st.session_state.pipeline = pipeline
        st.session_state.results = results
        st.session_state.annotated_video_path = str(output_dir / "annotated_match.mp4")
        if results is not None:
            st.session_state.processed = True
            progress_bar.progress(1.0)
            status_text.markdown(
                f'<div class="status-box">'
                f'<div style="display:flex; justify-content:space-between;">'
                f'<span style="color:#00FF00; font-weight:600;">Complete!</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.session_state.processing_error = pipeline.error or "Pipeline returned no results"
    except Exception as e:
        st.session_state.processing_error = f"{type(e).__name__}: {e}"

    st.session_state.processing = False
    st.rerun()

if st.session_state.processing_error:
    st.sidebar.markdown("---")
    st.sidebar.error(f"Processing failed: {st.session_state.processing_error}")

if st.session_state.processed and st.session_state.results:
    results = st.session_state.results
    pipeline = st.session_state.pipeline
    meta = results["video_metadata"]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">'
        f'<span style="color:#00FF00;">\u2705</span>'
        f'<span style="color:white; font-weight:600;">Analysis Complete</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'<div class="status-box">'
        f'<div style="display:flex; justify-content:space-between; margin-bottom:4px;">'
        f'<span style="color:#AAAAAA;">Duration</span>'
        f'<span style="color:white;">{meta.get("duration_sec", 0):.0f}s</span>'
        f'</div>'
        f'<div style="display:flex; justify-content:space-between; margin-bottom:4px;">'
        f'<span style="color:#AAAAAA;">Frames</span>'
        f'<span style="color:white;">{meta.get("processed_frames", 0)} / {meta.get("total_frames", 0)}</span>'
        f'</div>'
        f'<div style="display:flex; justify-content:space-between;">'
        f'<span style="color:#AAAAAA;">Players</span>'
        f'<span style="color:white;">{len(results.get("player_stats", {}))}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    export_dir = st.session_state.temp_dir / "analysis"
    json_path = export_dir / "analysis_results.json"
    if json_path.exists():
        with open(json_path, "rb") as f:
            st.sidebar.download_button(
                "Download Results (JSON)",
                data=f,
                file_name="tactical_analysis.json",
                mime="application/json",
                use_container_width=True,
            )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "\U0001f3c6 Overview", "\u26bd Tactical Pitch",
        "\U0001f3af Team Analysis", "\U0001f3c3 Player Analysis",
        "\U0001f4c8 Timeline",
    ])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Match Overview")
            st.markdown(
                f'<div style="display:flex; gap:12px; flex-wrap:wrap;">'
                f'<div class="card" style="flex:1; min-width:120px;">'
                f'<div class="card-label">Duration</div>'
                f'<div class="card-value">{meta.get("duration_sec", 0):.0f}s</div></div>'
                f'<div class="card" style="flex:1; min-width:120px;">'
                f'<div class="card-label">Players</div>'
                f'<div class="card-value">{len(results.get("player_stats", {}))}</div></div>'
                f'<div class="card" style="flex:1; min-width:120px;">'
                f'<div class="card-label">Formation A</div>'
                f'<div class="card-value">{results.get("dominant_formation_team_a", "?")}</div></div>'
                f'<div class="card" style="flex:1; min-width:120px;">'
                f'<div class="card-label">Formation B</div>'
                f'<div class="card-value">{results.get("dominant_formation_team_b", "?")}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.session_state.annotated_video_path and os.path.exists(st.session_state.annotated_video_path):
                st.subheader("Annotated Match Replay")
                with open(st.session_state.annotated_video_path, "rb") as f:
                    st.video(f.read())

        with col2:
            st.subheader("Team Metrics")
            ta = results.get("team_a_metrics", {})
            tb = results.get("team_b_metrics", {})
            team_metrics_df = pd.DataFrame({
                "Metric": ["Width", "Depth", "Compactness", "Players"],
                "Team A": [
                    f"{ta.get('width', 0):.1f}m", f"{ta.get('depth', 0):.1f}m",
                    f"{ta.get('compactness', 0):.1f}m", ta.get("num_players", 0),
                ],
                "Team B": [
                    f"{tb.get('width', 0):.1f}m", f"{tb.get('depth', 0):.1f}m",
                    f"{tb.get('compactness', 0):.1f}m", tb.get("num_players", 0),
                ],
            })
            st.dataframe(team_metrics_df, use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Ball Metrics")
            ball_m = results.get("ball_metrics", {})
            st.markdown(
                f'<div class="card"><div class="card-label">Avg Speed</div>'
                f'<div class="card-value">{ball_m.get("avg_speed", 0):.1f} m/s</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="card"><div class="card-label">Max Speed</div>'
                f'<div class="card-value">{ball_m.get("max_speed", 0):.1f} m/s</div></div>',
                unsafe_allow_html=True,
            )

    with tab2:
        st.subheader("Interactive Tactical Pitch")
        handle_tactical_pitch(results, pipeline)

    with tab3:
        st.subheader("Team Analysis")
        handle_team_analysis(results)

    with tab4:
        st.subheader("Player Analysis")
        handle_player_analysis(results)

    with tab5:
        st.subheader("Match Timeline & Formation History")
        handle_timeline(results, pipeline)

elif not st.session_state.processing:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f'<div style="text-align:center; padding:40px 20px;">'
            f'<div style="font-size:4rem; margin-bottom:16px;">\u26bd</div>'
            f'<h2>Ready to Analyze</h2>'
            f'<p style="color:#888; max-width:400px; margin:0 auto 24px;">'
            f'Upload a football match video and click <strong>Start Analysis</strong> '
            f'to begin the tactical analysis pipeline.</p>'
            f'<div style="display:flex; gap:20px; justify-content:center; flex-wrap:wrap;">'
            f'<div class="card" style="flex:1; min-width:140px; max-width:180px;">'
            f'<div style="font-size:1.5rem; margin-bottom:8px;">\U0001f9e0</div>'
            f'<div class="card-label">Detection</div>'
            f'<div style="color:#888; font-size:0.8rem;">YOLOv8</div></div>'
            f'<div class="card" style="flex:1; min-width:140px; max-width:180px;">'
            f'<div style="font-size:1.5rem; margin-bottom:8px;">\U0001f50d</div>'
            f'<div class="card-label">Tracking</div>'
            f'<div style="color:#888; font-size:0.8rem;">ByteTrack</div></div>'
            f'<div class="card" style="flex:1; min-width:140px; max-width:180px;">'
            f'<div style="font-size:1.5rem; margin-bottom:8px;">\U0001f3c8</div>'
            f'<div class="card-label">Formations</div>'
            f'<div style="color:#888; font-size:0.8rem;">K-Means</div></div>'
            f'<div class="card" style="flex:1; min-width:140px; max-width:180px;">'
            f'<div style="font-size:1.5rem; margin-bottom:8px;">\U0001f4ca</div>'
            f'<div class="card-label">Heatmaps</div>'
            f'<div style="color:#888; font-size:0.8rem;">Gaussian KDE</div></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    f"<div style='text-align:center; color:#555; font-size:0.75rem; padding:10px;'>"
    f"AI Tactical Analyst \u2022 Computer Vision & Sports Analytics Platform</div>",
    unsafe_allow_html=True,
)
