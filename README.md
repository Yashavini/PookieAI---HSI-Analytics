HSI-Llama: Hyperspectral Analytics
3D-CNN Pixel Classification + Llama-3.3 Intelligent QA
Project Link: https://github.com/Yashavini/Pookie-HSI-Analytics

📖 Overview
This project addresses the complexity of Hyperspectral Image (HSI) Analysis by combining deep learning with generative AI. Using the Indian Pines dataset, we implement a 3D-Convolutional Neural Network (3D-CNN) to identify land-cover classes with high precision. To make this data accessible, we integrated Llama-3.3 via Groq API, allowing users to ask natural language questions about the results through a premium, "Baddie Luxury" obsidian-themed interface.

✨ Key Features
Volumetric Feature Learning: Uses 3D-CNN to capture both spatial context and spectral signatures simultaneously.

Dimensionality Reduction: Implements PCA to compress 220 spectral bands into 30 principal components, optimizing performance.

Interactive LLM Layer: A conversational "Pookie AI" assistant that explains complex HSI statistics in simple terms.

High-End Visualization: Real-time generation of spectral fingerprints and RGB classification maps.

Luxury UI: Designed with a dark-mode obsidian aesthetic and gold accents for a professional conference-ready feel.

🛠️ Technical Implementation
The Pipeline
Preprocessing: Data normalization and PCA-based band reduction.

Patching: Creating 5×5×30 3D cubes for deep learning input.

Inference: The 3D-CNN classifies each pixel into one of 16 agricultural classes.

Reasoning: Results are summarized into JSON and interpreted by Llama-3.3-70B.

Tech Stack
Backend: Flask (Python 3.9+)

AI/ML: TensorFlow, NumPy, SciPy

LLM: Groq Cloud API (Llama-3.3)

Frontend: HTML5, CSS3, JavaScript (Fetch API)

🚦 Getting Started
1. Prerequisites
Ensure you have Python installed, then install the required libraries:

Bash
pip install -r requirements.txt
2. Setup API Key
For security, this repo does not contain my private API key.

Open app.py.

Locate the line: api_key = "YOUR_API_KEY_HERE".

Replace it with your actual Groq API Key.

3. Run the App
Bash
python app.py
Visit http://127.0.0.1:5000 to experience the luxury dashboard.

🎓 Academic Context
Developed as a 2nd Year CSE Mini-Project at JCT College of Engineering and Technology.

Author: Yashavini

Status: Active Development / Research Phase
