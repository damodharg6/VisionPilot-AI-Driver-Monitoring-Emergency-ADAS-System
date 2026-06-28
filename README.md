# VisionPilot-AI Driver Monitoring & Emergency ADAS System

![Developer](https://img.shields.io/badge/Developer-Giddaluru%20Damodhar-blue?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Python-3.x-green?style=for-the-badge&logo=python)
![Computer Vision](https://img.shields.io/badge/OpenCV-MediaPipe-orange?style=for-the-badge&logo=opencv)

Our system works as an AI-powered Driver Monitoring and Emergency ADAS Safety Platform designed to prevent accidents caused by driver fatigue or sleepiness. A mobile phone camera mounted like a dashboard camera continuously monitors the driver in real time using computer vision techniques. The system uses OpenCV and MediaPipe to detect the driver’s face, eyes, head position, and facial landmarks. By analyzing Eye Aspect Ratio (EAR), blink patterns, yawning behavior, and head tilt, the system continuously calculates the driver’s attention and fatigue level. If the driver shows signs of drowsiness, the system first issues progressive warnings through visual alerts and alarm sounds. If the driver still does not respond, the ADAS emergency system activates automatically. The vehicle simulation then uses virtual radar and sensor systems to analyze surrounding traffic, including vehicles in front, behind, and beside the car. Based on collision prediction, lane safety, and surrounding traffic conditions, the AI decision engine selects the safest maneuver path. The simulated autonomous system gradually reduces speed, performs smooth lane changes, avoids nearby vehicles, moves toward the emergency shoulder lane, and safely parks the vehicle. The entire system works in real time and demonstrates how modern automotive AI, driver monitoring, radar awareness, and autonomous emergency response systems can work together to improve road safety and prevent accidents.

## 📸 System in Action

### Normal Cruising (Driver Attentive)
![Driver Attentive](assets/demo_attentive.png)

### Emergency Safe-Stop (Driver Drowsy)
![Driver Drowsy](assets/demo_drowsy.png)

## 🚀 Key 

### 1. Advanced Driver Monitoring System (ADAS)
* **MediaPipe FaceLandmarker:** Employs advanced computer vision for highly robust, real-time tracking of 468 facial landmarks.
* **Adaptive EAR Calibration:** Eye Aspect Ratio (EAR) closure thresholds dynamically calibrate to the specific driver's face, replacing fixed/brittle values.
* **Multi-Factor Fatigue Scoring:** Calculates a continuous 0-100 Attention and Fatigue score by actively tracking eye closure duration, blink frequency, yawning patterns, head tilt (nodding), and signal stability.
* **Progressive Asynchronous Alerts:** Triggers visual and auditory warnings (soft warnings to loud alarms) asynchronously, ensuring the main camera and rendering loop never lag or drop frames.

### 2. Intelligent ADAS Decision Engine
* **Adaptive Cruise Control (ACC):** Actively measures the gap to the vehicle ahead using virtual radar and dynamically adjusts throttle to match speed and prevent rear-end collisions.
* **Sensor Fusion Threat Assessment:** Aggregates surrounding vehicles to calculate lane safety, obstacle density, blind-spot threats, and time-to-collision (TTC).
* **Emergency Autonomous Pullover:** If driver sleep is confirmed (Stage 4), the ADAS engine initiates an emergency takeover. It gradually reduces speed, uses the path planner to safely weave through adjacent lanes, and executes a diagonal park onto the emergency shoulder.
* **Vehicle Physics Engine:** Simulates realistic acceleration, braking inertia, and steering velocity. Lateral movement is bound to forward speed to ensure realistic cornering and lane changes.

### 3. High-Fidelity Highway Simulation
* **Dynamic AI Traffic:** Generates high-density, chaotic highway traffic (up to 45 simultaneous vehicles) including sedans, SUVs, trucks, and motorcycles.
* **Traffic Overtaking Logic:** AI vehicles utilize a following model and blocked-timers to actively seek gaps and overtake slower vehicles.
* **Cinematic 3D Perspective Renderer:** Maps the 2D logic grid into a visually stunning 3D perspective dashboard. Features dynamic road curvature, speed-synced dashed lines, vehicle yaw rotation, and functional hazard lights.
* **Integrated Telemetry HUD:** Displays real-time ADAS states, radar sweeps, and sensor fusion metrics in a beautifully stylized driver information panel.

## 📂 Project Architecture

```text
VisionPilot-AI/
├── main.py                     # Primary entry point & rendering loop
├── config.py                   # Global constants and simulation params
├── face_landmarker.task        # Pre-trained MediaPipe weights
├── driver_monitoring/
│   ├── face_detection.py       # Camera feed handling & MediaPipe setup
│   ├── fatigue_analysis.py     # Core logic for EAR/yawn tracking
│   ├── attention_scoring.py    # 0-100 temporal attention smoothing
│   └── alert_system.py         # Threaded audio warning triggers
├── adas/
│   ├── adas_engine.py          # Master orchestrator for autonomous logic
│   ├── path_planner.py         # Lane safety evaluation & route selection
│   ├── collision_prediction.py # TTC and danger threshold math
│   ├── sensor_fusion.py        # Environmental threat snapshotting
│   ├── autonomous_parking.py   # Safe-stop speed & shoulder logic
│   └── vehicle_physics.py      # Speed, acceleration, and steering inertia
├── simulation/
│   ├── highway_renderer.py     # 3D perspective engine & vehicle drawing
│   ├── traffic_ai.py           # High-density agent-based traffic AI
│   └── cinematic_effects.py    # UI/UX glow and boot sequence elements
└── assets/                     # Sounds, custom fonts, and UI themes
```

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.10+
* A connected Webcam

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Launch the System
```bash
python main.py
```

## 🎮 Controls

* **Q / ESC:** Quit application safely.
* **C:** Cycle camera source (useful if using DroidCam or external webcams).
* **Number keys:** Select specific camera on boot.

## 👨‍💻 Developer

Developed by **Giddaluru Damodhar**.

Dedicated to building intelligent AI solutions for a safer, autonomous automotive future.

*Disclaimer: This is simulation software designed for demonstration, research, and educational purposes.*
