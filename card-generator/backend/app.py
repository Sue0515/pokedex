
import requests
import io
import os 
import base64
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS

import card_generator

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

HUGGINGFACE_API_URL = os.environ.get("HUGGINGFACE_API_URL")
HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")

headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
}

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.json 

    # Extract parameters from request 
    prompt = data.get("prompt", "")
    negative_prompt = data.get("negative_prompt", "")
    num_inference_steps = data.get("num_inference_steps", 50)
    guidance_scale = data.get("guidance_scale", 7.5)
    height = data.get("height", 512)
    width = data.get("width", 512)

    # Prepare payload for Hugging Face API
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width
        }
    }

    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)

        if response.status_code==200:
            image_bytes = response.content 
            image = Image.open(io.BytesIO(image_bytes))

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return jsonify({
                "success": True,
                "image": f"data:image/png;base64,{img_str}"
            })
        
        else:
            return jsonify({
                "success": False,
                "error": f"API Error: {response.status_code}",
                "message": response.text
            }), 500
    
    except Exception as e:
         return jsonify({
            "success": False,
            "error": "Server Error",
            "message": str(e)
        }), 500



# @app.route("/create-card", methods=["POST"])
# def create_card():
#     data = request.json

#     try:




if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)