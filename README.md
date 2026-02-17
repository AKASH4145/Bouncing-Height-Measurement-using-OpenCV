#  Bouncing Height Estimation (OpenCV)

This project uses OpenCV and Python to estimate the bouncing height of a ball using webcam.  
It detects the ball in each frame, tracks its vertical position over time, and computes the relative bounce height using computer vision techniques.

The goal of this project is to demonstrate object detection, tracking, and motion analysis using OpenCV.

---

##  Features

-  Detects a bouncing ball from live webcam 
-  Tracks vertical motion frame-by-frame  
-  Estimates relative bounce height  
-  Simple computer vision pipeline   

---

##  Tech Stack

- Python
- OpenCV
- NumPy
- Matplotlib (for plotting, optional)

---

##  Project Structure

```text
bouncing-height/
├── main.py              # Main script
└── README.md
```
---
## Setup And Run

git clone https://github.com/AKASH4145/bouncing-height.git >>
cd bouncing-height >> 
python -m venv venv >> 
source venv/bin/activate (Linux / MacOS) or 
venv\Scripts\activate (Windows) >> 
pip install opencv-python numpy matplotlib >> 
python main.py

---
## Future Improvements

- Improve ball detection robustness under different lighting and backgrounds
- Visualize bounce height over time with interactive plots
- Track multiple objects simultaneously 
- Optimize performance for real-time processing 
---
## Author

Akash GS |
Mechanical Engineering student exploring 
AI, computer vision, and applied Python development