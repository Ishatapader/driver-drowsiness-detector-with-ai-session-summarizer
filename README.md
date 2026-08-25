# Driver Drowsiness Detection System with AI Session Summarizer
A webcam-based system that watches a driver's eyes, mouth, and head position in real time, detects signs of drowsiness, plays an alert, and logs every event for later review with an AI-generated session summary on top.
---
## Problem Statement
Driver fatigue is one of the leading causes of road accidents, and it's hard to catch in the moment drivers often don't realize their eyes have closed for a second too long or their head has started to drop until it's already a problem. Most vehicles have no built-in way to detect this in real time, and after a drive there's usually no data to look back on to understand when or how often it happened. This project closes that gap with a webcam-only monitoring solution that also turns the raw session data into a plain-language summary.
## Approach
- Extract facial landmarks per frame using MediaPipe's face mesh
- Compute EAR (eye closure), MAR (yawning), and head pitch from landmarks
- Estimate head pitch via `solvePnP`, compared to a calibrated baseline
- Require several consecutive threshold-breaching frames to filter false alerts
- Play alert on a separate thread; log episode start/end/duration to CSV
- Streamlit dashboard visualizes CSV data and generates an AI trip summary via Gemini
## Key Features
- Real-time face and posture tracking from a webcam
- Auto-calibration to the driver's normal sitting position
- False-alarm filtering using consecutive-frame checks
- Non-blocking audio alerts (runs on a separate thread, with device-loss recovery)
- Episode-based CSV logging (start/end + duration, not per-frame)
- Streamlit dashboard for reviewing past sessions
- AI-generated trip summary via Google Gemini
## Tech Stack
**Python3 | OpenCV | MediaPipe | NumPy | sounddevice | Streamlit | Pandas | Plotly | google-generativeai | python-dotenv**
## Project Structure
```
.
├── main.py                  # Webcam capture, landmark detection, alert logic, CSV logging
├── dashboard.py             # Streamlit dashboard + AI trip summary
├── config.json              # Externalized thresholds and runtime settings
├── requirements.txt         # Python dependencies
├── .env                     # GEMINI_API_KEY (not committed)
└── driver_telemetry.csv     # Auto-generated log of detected events
```
## Installation
**1. Clone the repository**
```bash
git clone https://github.com/Ishatapader/driver-drowsiness-detector-with-ai-session-summarizer.git
cd driver-drowsiness-detector-with-ai-session-summarizer
```
**2. Create a virtual environment (recommended)**
```bash
python3 -m venv venv
```
Activate it:
```bash
# Windows
venv\Scripts\activate

# Linux
source venv/bin/activate

# macOS
source venv/bin/activate
```
**3. Install dependencies**
```bash
pip install -r requirements.txt
```
**4. Add your Gemini API key**

Create a `.env` file in the project root:
```
GEMINI_API_KEY="your-gemini-api-key-here"
```
## Usage
**Run the monitoring system:**
```bash
python3 main.py
```
Sit normally during the brief calibration phase at startup. Press `m` to mute/unmute the alert, `q` to quit.

**Review session data:**
```bash
streamlit run dashboard.py
```
Opens a browser dashboard showing total warnings, time-series charts for each metric, and an on-demand AI trip summary.
---

## Live Demo
🚀 Try the dashboard here: https://driver-drowsiness-detector-with-ai-session-summarizer-wgztbz6p.streamlit.app/

## Author
**Isha Tapader**

🔗 GitHub: [github.com/Ishatapader](https://github.com/Ishatapader)

💼 LinkedIn: [linkedin.com/in/isha-tapader-116680247](https://www.linkedin.com/in/isha-tapader-116680247/)
