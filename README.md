# Multi-Robot Cooperative Tracking System (LIMO Robots)

This repository contains the implementation of a multi-robot cooperative tracking system developed for the Intelligent Distributed Systems course.

The goal is to enable a team of mobile robots (LIMO) to collaboratively localize and track a moving human agent in a shared environment while maintaining a coordinated formation around the target. The system combines decentralized estimation, vision-based perception, and model-based control.

---

## 🚀 System Overview

The system consists of three LIMO mobile robots that cooperatively track a moving person while maintaining a coordinated formation around the target. The robots combine vision-based perception, cooperative state estimation, and model-based control to ensure consistent relative positioning with respect to the human agent.

Each robot operates with local sensing and contributes to a shared estimation process through inter-agent measurements.

---

## 🧠 Core Methods

### 🔹 Cooperative State Estimation
- Implemented an **Interacting Multiple Model / decentralized EKF-based cooperative localization framework**
- Enables **inter-robot relative measurements**
- Fusion of:
  - wheel odometry
  - stereo camera measurements
  - relative observations between agents

This allows consistent multi-agent localization even under partial observability.

---

### 🔹 Perception (Vision)
- **YOLO-based detection** for human tracking
- Marker-based detection for inter-robot observations
- Multi-source visual fusion for robust target identification

---

### 🔹 Control
- **Model Predictive Control (MPC)** for trajectory tracking and formation maintenance
- Each robot computes local control actions to:
  - track the moving target
  - preserve formation constraints relative to other agents
  - ensure smooth coordination in dynamic environments

---

## 🤖 System Architecture

Each LIMO robot runs a local pipeline:

1. Perception:
   - YOLO detection (human)
   - Marker detection (other robots)

2. State Estimation:
   - Cooperative EKF with inter-agent measurements
   - Fusion of odometry + stereo vision + relative observations

3. Control:
   - MPC-based motion controller
   - local trajectory generation for tracking and formation keeping

Communication between agents is handled using ROS2 topics.

---

## 🛠️ Technologies

- Python
- ROS2
- OpenCV
- YOLO (object detection)
- NumPy
- Nonlinear estimation (EKF)
- Model Predictive Control (MPC)
- Stereo vision

---

## 📊 Key Features

- Decentralized multi-robot state estimation
- Cooperative localization using inter-agent measurements
- Vision-based human tracking (YOLO)
- Marker-based robot-to-robot detection
- MPC-based control in dynamic environments
- Formation-based multi-robot tracking of a moving human
- Simulation-first → real robot deployment pipeline

---

## 🎯 Objectives

- Maintain a dynamic formation of robots around a moving human target
- Achieve robust multi-robot tracking of a moving human
- Maintain consistent localization under uncertainty
- Leverage inter-robot cooperation to improve estimation accuracy
- Transition from simulation to real-world robotic deployment

---

## 👤 Author

Marco Misseroni and Federico Battisti  
MSc Mechatronics Engineering – Electronics and Robotics  
University of Trento  
