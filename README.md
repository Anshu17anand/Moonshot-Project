# Vertebral Body Tumor Detection & AI-Assisted Robotic Control

Demo video link: https://drive.google.com/file/d/1-QBQygtlptnu4bQR8nQaM2cqHSKl4k4H/view?usp=sharing

**AI-Assisted Surgical Robotics for Endoscopic Spine Surgery **

A computer vision–driven system that detects vertebral body tumors in real time from an endoscopic camera feed and assists robotic navigation using **voice commands, controller input, and an offline AI assistant**.

---

## 📌 Overview

During endoscopic spine surgery, surgeons operate through extremely small incisions with limited visual feedback and almost no depth perception. Achieving sub-millimeter precision while removing vertebral body tumors is cognitively demanding and error-prone.

This project explores how **computer vision, voice interfaces, and local AI models** can be combined to:
- Detect spinal tumors in real time
- Provide objective spatial guidance
- Reduce manual intervention during robotic navigation

The system integrates **tumor detection**, **voice-controlled movement**, and an **AI assistant** to augment surgical decision-making.

---

## 🚀 Key Features

- **Real-time tumor detection**
  - Detects vertebral tumors as orange-marked regions
  - Displays confidence score for each detection
- **Distance estimation**
  - Approximates distance to tumor (in millimeters)
  - Helps guide centering and approach
- **Voice-controlled robot navigation**
  - Offline speech-to-text using Vosk
- **AI-assisted guidance (Jarvis)**
  - Offline AI assistant (via Ollama)
  - Can answer queries such as tumor confidence, distance, centering, and movement guidance
- **Fully offline operation**
  - No internet required during runtime
  - No token limits

---

## 🧠 System Architecture

1. **Camera feed** from robot-mounted endoscope  
2. **Computer vision pipeline**  
   - Frame capture → detection → confidence + distance estimation  
3. **Control layer**  
   - Voice commands (STT)  
   - Controller input  
4. **Robot communication**  
   - UDP commands sent to robot brick  
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
- Raspberry Pi (robot control brick; physical robot not yet built)

---

## 📂 Project Structure

```text
├── working.py                  # Main integrated pipeline
├── working_robot_stable.py     # Stable robot control version
├── enhanced.py                 # Experimental enhancements
├── udp_test.py                 # UDP communication testing
├── surgery_report.json         # Logged surgical metrics
├── vosk_models/                # Offline speech-to-text models
├── .matplotlib/                # Matplotlib cache
├── __pycache__/                # Python cache
└── README.md
```

---

## 🌱 Future Scope

- Accurate depth estimation using stereo or depth sensors  
- Closed-loop robotic feedback  
- Higher-precision tumor segmentation  
- Surgeon-specific calibration  
- Integration with real surgical robotic arms  
- Research-grade clinical validation  

---

## 🎥 Demo

📹 **Demo video & screenshots:**  
<img src="moonshot1.png" width="800" />
<img src="moonshot2.png" width="800" />



