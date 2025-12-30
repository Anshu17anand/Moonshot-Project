🦴 Vertebral Body Tumor Detection & AI-Assisted Robotic Control

AI-Assisted Surgical Robotics for Endoscopic Spine Surgery (Prototype)

A computer vision–driven system that detects vertebral body tumors in real time from an endoscopic camera feed and assists robotic navigation using voice commands, controller input, and an offline AI assistant.

⚠️ Research & educational prototype only. Not a medical device.

📌 Overview

During endoscopic spine surgery, surgeons operate through extremely small incisions with limited visual feedback and almost no depth perception. Achieving sub-millimeter precision while removing vertebral body tumors is cognitively demanding and error-prone.

This project explores how computer vision, voice interfaces, and local AI models can be combined to:

Detect spinal tumors in real time

Provide objective spatial guidance

Reduce manual intervention during robotic navigation

The system integrates tumor detection, voice-controlled movement, and an AI assistant to augment surgical decision-making.

🚀 Key Features

Real-time tumor detection

Detects vertebral tumors as orange-marked regions

Displays confidence score for each detection

Distance estimation

Approximates distance to tumor (in millimeters)

Helps guide centering and approach

Voice-controlled robot navigation

Offline speech-to-text using Vosk

Commands: up, down, left, right, stop

Controller-based control

PS4 controller support for manual override

AI-assisted guidance (Jarvis)

Offline AI assistant (via Ollama)

Can answer queries such as:

Tumor confidence

Distance to tumor

Whether the tumor is centered

How to move to align with the tumor

Fully offline operation

No internet required during runtime

No token limits

🧠 System Architecture

Camera feed from robot-mounted endoscope

Computer vision pipeline

Frame capture → detection → confidence + distance estimation

Control layer

Voice commands (STT)

PS4 controller input

Robot communication

UDP commands sent to robot brick

AI assistant

Local LLM for contextual surgical guidance

🛠️ Tech Stack
Languages & Core Libraries

Python

OpenCV – real-time image processing

NumPy – numerical computations

Matplotlib – visualizations & debugging

AI & ML

Roboflow

Dataset annotation

Model training for tumor detection

Ollama

Offline local LLM (Jarvis assistant)

Vosk

Offline speech-to-text (voice control)

Systems & Infrastructure

Docker – local inference & reproducibility

UDP sockets – robot communication

Raspberry Pi – robot control brick (hardware abstraction; physical robot not built yet)
