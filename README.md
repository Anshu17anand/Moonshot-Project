# Vertebral Body Tumor Detection & AI-Assisted Robotic Control

Demo video link: https://drive.google.com/file/d/1-QBQygtlptnu4bQR8nQaM2cqHSKl4k4H/view?usp=sharing

**AI-Assisted Surgical Robotics for Endoscopic Spine Surgery**

A computer vision–driven system that detects vertebral body tumors in real time from an endoscopic camera feed and assists robotic navigation using **voice commands, controller input, and offline AI models**. This proof-of-concept was developed on a da Vinci robot replica to demonstrate how AI can augment surgical precision in minimally invasive procedures.

---

## 📌 Overview

During endoscopic spine surgery, surgeons operate through extremely small incisions with limited visual feedback and almost no depth perception. Achieving sub-millimeter precision while removing vertebral tumors is challenging and requires constant manual adjustment and spatial reasoning.

This project explores how **computer vision, voice interfaces, and local AI models** can be combined to:
- Detect spinal tumors in real time with confidence scoring
- Provide objective spatial guidance through distance estimation
- Reduce manual intervention during robotic navigation

The system integrates **tumor detection**, **voice-controlled movement**, and an **AI assistant** to augment surgical decision-making. All processing runs locally without internet connectivity, making it suitable for sterile surgical environments.

---

## 🚀 Key Features

- **Real-time tumor detection**
  - Detects vertebral tumors as orange-marked regions in endoscopic feed
  - Displays confidence score for each detection
- **Distance estimation**
  - Approximates distance to tumor (in millimeters)
  - Helps guide centering and approach vectors
- **Voice-controlled robot navigation**
  - Offline speech-to-text using Vosk
  - Natural language command processing
- **AI-assisted guidance (Jarvis)**
  - Offline AI assistant (via Ollama)
  - Can answer queries such as tumor confidence, distance, centering adjustments, and movement guidance
- **Fully offline operation**
  - No internet required during runtime
  - No token limits or cloud dependencies
  - Suitable for OR environments

---

## 🧠 System Architecture

1. **Camera feed** from robot-mounted endoscope  
2. **Computer vision pipeline**  
   - Frame capture → detection → confidence + distance estimation  
3. **Control layer**  
   - Voice commands (STT via Vosk)  
   - Controller input handling  
4. **Robot communication**  
   - UDP commands sent to robot control brick (Raspberry Pi)
5. **AI assistant**  
   - Local LLM for contextual surgical guidance  

---

## 🛠️ Tech Stack

### Languages & Core Libraries
- Python
- OpenCV
- NumPy
- Matplotlib

### AI & ML
- Roboflow (dataset annotation & model training)
- Ollama (offline local LLM)
- Vosk (offline speech-to-text)

### Systems & Infrastructure
- Docker (local inference & reproducibility)
- UDP sockets (robot communication)
- Raspberry Pi (robot control brick; custom da Vinci replica)

---

## 📂 Project Structure

```text
├── working_robot_stable.py     # Main entry point - Stable robot control version
├── working.py                  # Integrated pipeline (experimental)
├── enhanced.py                 # Experimental enhancements
├── udp_test.py                 # UDP communication testing
├── surgery_report.json         # Logged surgical metrics
├── vosk_models/                # Offline speech-to-text models
├── .matplotlib/                # Matplotlib cache
├── __pycache__/                # Python cache
└── README.md
```

---

## ▶️ Running the System

The main entry point is `working_robot_stable.py`, which runs the complete pipeline with robot control:

```bash
python working_robot_stable.py
```

This script:
- Initializes the endoscopic camera feed
- Loads pre-trained tumor detection models (via Roboflow)
- Activates voice command listening (Vosk)
- Connects to the robot control brick via UDP
- Enables the Jarvis AI assistant for contextual guidance

---

## 📋 Project Status

**Status:** Proof-of-Concept (PoC)  
**Scale:** Small-scale da Vinci robot replica (non-commercial, research setup)  
**Hardware:** Third-party surgical robot replica with Raspberry Pi control brick  
**Current Capability:** Real-time tumor detection + voice-guided robot control in lab environment

---

## 🌱 Future Scope

- Accurate depth estimation using stereo or depth sensors  
- Closed-loop robotic feedback integration
- Higher-precision tumor segmentation with advanced model architectures
- Surgeon-specific calibration & personalization
- Integration with commercial surgical robotic arms (da Vinci, Raven II, etc.)
- Research-grade clinical validation on cadaver models
- Real-time image registration for spatial consistency

---

## 🎥 Demo

📹 **Demo screenshots from proof-of-concept testing:**  
<img src="moonshot1.png" width="800" alt="Tumor detection with confidence scoring" />
<img src="moonshot2.png" width="800" alt="AI-assisted navigation interface" />

---

## ⚖️ License

[Add your license here - e.g., MIT, Apache 2.0, etc.]

---

## 📚 Acknowledgments

- **Robot Hardware:** Third-party da Vinci replica for research purposes
- **Dataset & Model Training:** Roboflow annotation platform
- **AI & ML:** Ollama for offline LLM, Vosk for speech recognition
- **Inspiration:** Research in AI-assisted minimally invasive surgery

