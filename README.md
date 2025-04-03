# Gaze Guard

## Description
A real-time webcam application using OpenCV that detects if you are looking away from your screen for too long, helping maintain focus during tasks.

## Features
- Real-time eye tracking using Haar Cascades or a similar lightweight model.
- Detection of gaze direction (simplistic: eyes visible vs. eyes not clearly visible/looking sideways).
- User-configurable timer to set the maximum allowed duration for looking away.
- Visual or audible alert when the 'away-gaze' timer expires.
- Simple calibration or ROI selection for better eye detection.

## Learning Benefits
Gain experience with real-time video capture, facial landmark detection (specifically eyes), implementing simple state machines (tracking gaze over time), applying basic computer vision techniques for a practical productivity tool, and working with OpenCV Haar Cascades or dlib.

## Technologies Used
- opencv-python
- numpy
- dlib (optional, for potentially better landmark detection)
- playsound (optional, for audio alerts)

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gaze-guard.git
cd gaze-guard

# Install dependencies
pip install -r requirements.txt
```

## Usage
[Instructions on how to use the project]

## Project Structure
[Brief explanation of the project structure]

## License
MIT

## Created with AI
This project was automatically generated using an AI-powered project generator.
