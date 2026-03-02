mport os
import requests
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
from scipy.io import loadmat
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- HSI ANALYTICS ENGINE ---
app = Flask(__name__)
CORS(app)

# Credentials & Model Config
GROQ_API_KEY = "YOUR_API_KEY_HERE"
GROQ_URL = "ENTER_YOUR_API_URL_HERE"
LUXURY_MODEL = "llama-3.3-70b-versatile"

# Hardcoded Luxury Paths (Matching your structure)
CORRECTED_PATH = r"C:\Users\ADMIN\Desktop\Mini Project\Data\Indian_pines_corrected.mat"
GT_PATH = r"C:\Users\ADMIN\Desktop\Mini Project\Data\Indian_pines_gt.mat"

def analyze_hsi_metadata():
    """Extracts high-end technical metadata from the .mat files."""
    try:
        # Loading the Indian Pines Cube
        data_cube = loadmat(r"C:\Users\ADMIN\Desktop\Mini Project\Data\Indian_pines_corrected.mat")['indian_pines_corrected']
        ground_truth = loadmat(r"C:\Users\ADMIN\Desktop\Mini Project\Data\Indian_pines_gt.mat")['indian_pines_gt']
        
        h, w, b = data_cube.shape
        classes = len(np.unique(ground_truth)) - 1 # Background removal
        
        return {
            "dimensions": f"{h}x{w}",
            "bands": b,
            "classes": classes,
            "status": "Loaded Successfully"
        }
    except Exception as e:
        print(f"Metadata Error: {e}")
        return {"status": "Offline", "error": str(e)}

# --- NEW LUXURY DATA VISUALIZATION ENDPOINTS ---

@app.route("/api/visualize/rgb", methods=["GET"])
def get_rgb_image():
    """Generates a luxury RGB preview for the 'Data' icon."""
    try:
        data_cube = loadmat(CORRECTED_PATH)['indian_pines_corrected']
        # Select bands for a standard RGB representation (Bands 29, 19, 9)
        rgb = data_cube[:, :, [29, 19, 9]].astype(float)
        rgb /= np.max(rgb) # Luxury normalization
        
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
    """Generates a gold spectral graph for the 'Analytics' icon."""
    try:
        data_cube = loadmat(CORRECTED_PATH)['indian_pines_corrected']
        pixel_sig = data_cube[50, 50, :] # Extracting specific spectral signature
        
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 4))
        plt.plot(pixel_sig, color='#d4af37', linewidth=2) # Luxury Gold Line
        plt.title("Spectral Signature: Pixel (50, 50)", color='#d4af37')
        plt.xlabel("Band Number", color='#666')
        plt.ylabel("Intensity", color='#666')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True)
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return jsonify({"graph": graph_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ORIGINAL CHAT LOGIC (PROTECTED) ---

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_input = data.get('question', '')
        meta = analyze_hsi_metadata()
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        system_instructions = (
            f"You are Pookie, a world-class AI research assistant specialized in Deep Learning and HSI. "
            f"You are currently helping Students with Yash's prestigious Indian Pines project in Conferrance hall. "
            f"Dataset Intel: The cube has {meta.get('bands')} spectral bands and {meta.get('classes')} unique land-cover classes. "
            "Your tone is elegant, ambitious, and highly technical. Use professional language with a touch of warmth, No extras; Speak short and sweet. You can also use descent GenZ words like slay, diva etc.. "
            "Call them Babe or ocassionally boss babe or diva and ensure they feels like a top-tier CSE researchers."
        )

        payload = {
            "model": LUXURY_MODEL,
            "messages": [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.5,
            "max_tokens": 1024
        }

        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            return jsonify({
                "answer": answer,
                "metadata": meta
            })
        else:
            error_msg = response.json().get('error', {}).get('message', 'Network Glitch')
            return jsonify({"answer": f"Pookie encountered a slight delay: {error_msg}. 🎀"}), 500

    except Exception as e:
        return jsonify({"answer": f"Grok system maintenance required: {str(e)}"}), 500

if __name__ == "__main__":
    print("\n" + "✧"*30)
    print("  HYPERSPECTRAL ANALYTICS IS LIVE  ")
    print(f"  MODE: LUXURY | TARGET: YASH")
    print("✧"*30 + "\n")
    app.run(debug=True, port=5000)