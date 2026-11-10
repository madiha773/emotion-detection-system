import cv2
import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Space+Mono&display=swap');

  html, body, [class*="css"] {
    background-color: #0d0d0d;
    color: #e8e8e8;
    font-family: 'Space Grotesk', sans-serif;
  }

  .title-block {
    border-left: 4px solid #ff3c3c;
    padding: 6px 0 6px 16px;
    margin-bottom: 24px;
  }
  .title-block h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
    color: #ffffff;
  }
  .title-block p {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #666;
    margin: 4px 0 0 0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* dominant emotion card */
  .emotion-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }
  .emotion-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
  }
  .emotion-card .value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ff3c3c;
    letter-spacing: -1px;
  }
  .emotion-card .faces {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #444;
    margin-top: 4px;
  }

  /* emotion bar rows */
  .bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .bar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #888;
    width: 68px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    flex-shrink: 0;
  }
  .bar-track {
    flex: 1;
    background: #1e1e1e;
    border-radius: 2px;
    height: 8px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
  }
  .bar-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #555;
    width: 40px;
    text-align: right;
    flex-shrink: 0;
  }

  /* status pill */
  .status-pill {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 20px;
    margin-bottom: 16px;
  }
  .status-live   { background: #1a2e1a; color: #4cff4c; border: 1px solid #2a4a2a; }
  .status-idle   { background: #1e1e1e; color: #555;    border: 1px solid #2a2a2a; }

  /* hide default streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }

  /* override streamlit button */
  div[data-testid="stButton"] > button {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    border-radius: 4px;
    border: none;
    padding: 10px 28px;
    cursor: pointer;
    letter-spacing: 0.03em;
    transition: opacity 0.15s;
  }
  div[data-testid="stButton"] > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── Emotion colours (bar fill) ────────────────────────────────────────────────
EMOTION_COLORS = {
    "happy":    "#f5c542",
    "neutral":  "#7ec8e3",
    "sad":      "#6ca0dc",
    "angry":    "#ff3c3c",
    "fear":     "#b57bee",
    "surprise": "#ff9a3c",
    "disgust":  "#6bcf7f",
}
EMOTION_ORDER = ["happy", "neutral", "sad", "angry", "fear", "surprise", "disgust"]

# ── Session state ─────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False

# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>🎭 Emotion Detector</h1>
  <p>Real-time facial expression analysis · DeepFace</p>
</div>
""", unsafe_allow_html=True)

col_feed, col_panel = st.columns([3, 2], gap="large")

with col_feed:
    # status pill placeholder
    status_ph = st.empty()
    # webcam frame placeholder
    frame_ph  = st.empty()

with col_panel:
    # Start / Stop buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start_clicked = st.button("▶  Start", use_container_width=True, type="primary")
    with btn_col2:
        stop_clicked  = st.button("⏹  Stop",  use_container_width=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # dominant emotion card
    dominant_ph = st.empty()
    # emotion bars
    bars_ph     = st.empty()

# ── Button logic ──────────────────────────────────────────────────────────────
if start_clicked:
    st.session_state.running = True
if stop_clicked:
    st.session_state.running = False

# ── Helper: render emotion bars '"���d�0V�z��
�T}s�%�a}�&��───────────────────────────────
def bars_html(scores: dict) -> str:
    rows = ""
    for emo in EMOTION_ORDER:
        pct   = scores.get(emo, 0.0)
        color = EMOTION_COLORS.get(emo, "#888")
        rows += f"""
        <div class="bar-row">
          <span class="bar-label">{emo}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
          </div>
          <span class="bar-pct">{pct:.1f}%</span>
        </div>"""
    return f'<div style="padding:4px 0">{rows}</div>'

# ── Helper: render dominant emotion card HTML ─────────────────────────────────
def dominant_html(emotion: str, face_count: int) -> str:
    return f"""
    <div class="emotion-card">
      <div class="label">Dominant emotion</div>
      <div class="value">{emotion.upper()}</div>
      <div class="faces">{'👤 ' * face_count}{face_count} face{'s' if face_count != 1 else ''} detected</div>
    </div>"""

# ── Idle state render ─────────────────────────────────────────────────────────
if not st.session_state.running:
  '"���d�0V�z��
�T}s�%�a}�&��="status-pill status-idle">● Idle — press Start</span>',
                       unsafe_allow_html=True)
    frame_ph.markdown("""
    <div style="background:#111;border:1px solid #1e1e1e;border-radius:6px;
                height:360px;display:flex;align-items:center;justify-content:center;
                color:#333;font-family:'Space Mono',monospace;font-size:0.8rem;">
      webcam feed will appear here
    </div>""", unsafe_allow_html=True)
    dominant_ph.markdown(dominant_html("—", 0), unsafe_allow_html=True)
    bars_ph.markdown(bars_html({e: 0 for e in EMOTION_ORDER}), unsafe_allow_html=True)

# ── Live detection loop ───────────────────────────────────────────────────────
if st.session_state.running:
    status_ph.markdown('<span class="status-pill status-live">● Live</span>',
                       unsafe_allow_html=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Webcam not found. Check it's connected and not used by another app.")
        st.session_state.running = False
        st.stop()

    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Could not read frame.")
                break

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )

            dominant_emotion = "—"
            all_scores       = {e: 0.0 for e in EMOTION_ORDER}

            for idx, (x, y, w, h) in enumerate(faces):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_roi  = rgb_frame[y:y+h, x:x+w]

                try:
                    result   = DeepFace.analyze(face_roi, actions=["emotion"],
                                                enforce_detection=False, silent=True)
                    emotion  = result[0]["dominant_emotion"]
                    scores   = result[0]["emotion"]

                    # use first face's scores for panel
                    if idx == 0:
                        dominant_emotion = emotion
                        all_scores       = {k.lower(): v for k, v in scores.items()}

                    # draw box + label
                    color = (255, 60, 60)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, emotion.upper(), (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                except Exception:
                    pass

            # show frame
            display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_ph.image(display_rgb, channels="RGB", use_container_width=True)

            # update panel
           '"���d�0V�z��
�T}s�%�a}�&��ml(dominant_emotion, len(faces)),
                                  unsafe_allow_html=True)
            bars_ph.markdown(bars_html(all_scores), unsafe_allow_html=True)

            time.sleep(0.04)   # ~25 fps

    finally:
        cap.release()
