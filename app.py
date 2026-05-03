import os
import requests
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
from scipy.io import loadmat
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Credentials from Environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
LUXURY_MODEL = "llama-3.3-70b-versatile"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORRECTED_PATH = os.path.join(BASE_DIR, "Indian_pines_corrected.mat")
GT_PATH = os.path.join(BASE_DIR, "Indian_pines_gt.mat")

# --- DATA PRE-LOADING (The Speed Fix) ---
# We load the data once globally so the routes respond instantly
print("Professor Pookie is loading the Indian Pines data... stay slay.")
try:
    if os.path.exists(CORRECTED_PATH):
        DATA_CUBE = loadmat(CORRECTED_PATH)['indian_pines_corrected']
        GROUND_TRUTH = loadmat(GT_PATH)['indian_pines_gt']
        DATA_LOADED = True
    else:
        DATA_LOADED = False
        print("Error: Dataset files not found in root directory.")
except Exception as e:
    DATA_LOADED = False
    print(f"Error loading dataset: {e}")

@app.route('/')
def home():
    return "<h1>Pookie AI Backend is Online, Diva! </h1><p>The HSI Research Portal is ready.</p>"

def analyze_hsi_metadata():
    """Extracts technical metadata from pre-loaded data."""
    if not DATA_LOADED:
        return {"status": "Offline", "error": "Dataset not loaded."}
    try:
        h, w, b = DATA_CUBE.shape
        classes = len(np.unique(GROUND_TRUTH)) - 1
        return {
            "dimensions": f"{h}x{w}",
            "bands": b,
            "classes": classes,
            "status": "Loaded Successfully"
        }
    except Exception as e:
        return {"status": "Error", "error": str(e)}

@app.route("/api/visualize/rgb", methods=["GET"])
def get_rgb_image():
    if not DATA_LOADED:
        return jsonify({"error": "Dataset not loaded on server"}), 500
    try:
        # Using the pre-loaded DATA_CUBE for instant response
        rgb = DATA_CUBE[:, :, [29, 19, 9]].astype(float)
        rgb /= np.max(rgb)
        plt.figure(figsize=(5, 5), facecolor='black')
        plt.imshow(rgb)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return jsonify({"image": img_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/visualize/spectral", methods=["GET"])
def get_spectral_graph():
    if not DATA_LOADED:
        return jsonify({"error": "Dataset not loaded on server"}), 500
    try:
        # Instant access to spectral bands
        pixel_sig = DATA_CUBE[50, 50, :]
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 4))
        plt.plot(pixel_sig, color='#d4af37', linewidth=2)
        plt.title("Spectral Signature: Pixel (50, 50)", color='#d4af37')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True)
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return jsonify({"graph": graph_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        meta = analyze_hsi_metadata()
        user_input = request.json.get('question', '')
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        system_instructions = (
          f"You are Professor Pookie, PhD mentor for Indian Pines. "
          f"Dataset Status: {meta.get('bands')} bands of pure data. "
           "Tone: Academic Elite meets GenZ. You are highly technical but use words like 'slay', 'main character energy', and 'diva'. "
           "Keep responses very short and sweet, and intellectually stimulating. "
           "Call them 'Boss Babe' or 'Scholar Diva'. No yap, just facts and vibes."
        )

        payload = {
            "model": LUXURY_MODEL,
            "messages": [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.3, # Lowered temperature slightly for faster logic
            "max_tokens": 150    # Added token limit to stop AI from 'yapping' too long
        }

        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content']
            return jsonify({"answer": answer, "metadata": meta})
        return jsonify({"answer": "Pookie is recharging. Check API key, Diva. 🎀"}), 500
    except Exception as e:
        return jsonify({"answer": f"Glitch: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
