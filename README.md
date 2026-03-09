HSI Analytics: Indian Pines Research Portal
Deep Learning for Hyperspectral Image Classification
💎 Project Overview
This project is an elite research dashboard designed for the analysis and classification of the Indian Pines Hyperspectral Dataset. It features a custom 3D-CNN/HybridSN architecture for spatial-spectral feature extraction and a GenZ-inspired AI Research Mentor, Professor Pookie, powered by Llama 3.3 via Groq.

✨ Key Features
Live Spectral Analysis: Real-time extraction of spectral signatures from specific pixel coordinates.

RGB Reconstruction: Visualization of high-dimensional HSI data into human-readable RGB format.

AI Mentorship: Integrated LLM for technical guidance on HSI processing and model metrics.

Luxury UI: Glassmorphism-based dashboard designed for high-end research presentation.

📂 Project Structure
Plaintext
/HSI-Analytics
│   requirements.txt      # Project dependencies
│   Procfile              # Deployment config for Cloud (Render)
│   README.md             # You are here!
│
├───Data/                 # .mat Dataset storage
│       Indian_pines_corrected.mat
│       Indian_pines_gt.mat
│
├───Backend/              # Flask Server Logic
│       app.py            # Main API & AI Integration
│       model.py          # Deep Learning Model (3D-CNN)
│       util.py           # Data processing utilities
│       .env              # API Credentials (Hidden)
│
└───Frontend/             # Dashboard Interface
        index.html        # Glassmorphism UI
🚀 Setup & Installation
1. Clone the Repository
Bash
git clone https://github.com/YOUR_USERNAME/HSI-Analytics.git
cd HSI-Analytics
2. Configure Environment
Create a .env file in the Backend/ directory:

Plaintext
GROQ_API_KEY=your_actual_key_here
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Run the Engine
Bash
python Backend/app.py
Open Frontend/index.html in your browser. Access Code: Password

🧠 Model Architecture
The system utilizes a Spatial-Spectral 3D Convolutional Neural Network.

3D Convolutions: Captures the spectral correlations across neighboring bands.

2D Convolutions: Focuses on spatial patterns within the land-cover classes.

Softmax Layer: Classifies the 16 distinct land-cover categories of Indian Pines.


Yashavini
2nd Year CSE


Would you like me to help you write the specific Git commands to upload your folders to GitHub for the first time?
