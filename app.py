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

load_dotenv()
app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
LUXURY_MODEL = "llama-3.3-70b-versatile"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORRECTED_PATH = os.path.join(BASE_DIR, "Indian_pines_corrected.mat")
GT_PATH = os.path.join(BASE_DIR, "Indian_pines_gt.mat")

# --- LIGHTWEIGHT DATA LOADING ---
# Instead of the whole cube, we only pre-process the tiny images we need.
print("Scholar Diva, I'm preparing the lightweight visuals...")

def get_prepped_data():
    if not os.path.exists(CORRECTED_PATH):
        return None, None, "Offline"
    
    # Load data
    raw_data = loadmat(CORRECTED_PATH)['indian_pines_corrected']
    gt_data = loadmat(GT_PATH)['indian_pines_gt']
    
    # 1. Create a tiny RGB image once (Bands 29, 19, 9)
    rgb = raw_data[:, :, [29, 19, 9]].astype(float)
    rgb /= np.max(rgb)
    
    # 2. Get one signature once
    sig = raw_data[50, 50, :].tolist()
    
    # 3. Get metadata
    meta = {
        "dimensions": f"{raw_data.shape[0]}x{raw_data.shape[1]}",
        "bands": raw_data.shape[2],
        "classes": len(np.unique(gt_data)) - 1,
        "status": "Slaying"
    }
    
    # DELETE raw_data from memory immediately to save Render RAM
    del raw_data
    return rgb, sig, meta

# Load only the results into memory
PREPPED_RGB, PREPPED_SIG, HSI_META = get_prepped_data()

@app.route('/')
def home():
    return "<h1>Pookie AI is Lean and Fast, Diva!</h1>"

@app.route("/api/visualize/rgb", methods=["GET"])
def get_rgb_image():
    try:
        plt.figure(figsize=(5, 5), facecolor='black')
        plt.imshow(PREPPED_RGB)
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
    try:
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 4))
        plt.plot(PREPPED_SIG, color='#d4af37', linewidth=2)
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
        user_input = request.json.get('question', '')
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        
        system_instructions = (
            f"You are Professor Pookie. Dataset: {HSI_META['bands']} bands. "
            "Tone: Academic Elite meets GenZ. Short, sweet, high-vibes. "
            "Call them 'Scholar Diva'. No yap."
        )

        payload = {
            "model": LUXURY_MODEL,
            "messages": [{"role": "system", "content": system_instructions}, {"role": "user", "content": user_input}],
            "temperature": 0.2,
            "max_tokens": 100
        }

        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=10)
        answer = response.json()['choices'][0]['message']['content']
        return jsonify({"answer": answer, "metadata": HSI_META})
    except:
        return jsonify({"answer": "Network glitch, Diva. Stay slay."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
