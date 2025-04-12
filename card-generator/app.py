import io
import os
import base64
import textwrap
import numpy as np
import tempfile
import time
import json
import re
import torch
import streamlit as st

from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, Dict, List
from diffusers import StableDiffusionPipeline

from custom_templates import(
    create_pokemon_card, CARD_STYLE_TEMPLATES, get_preview_images, get_image_base64
)

# Set page config
st.set_page_config(
    page_title = "Pokémon Card Generator",
    page_icon="🎴",
    layout="wide"
)

# Constants and settings
CARD_TYPES = ["Normal", "Fire", "Water", "Grass", "Electric", "Psychic", "Fighting", "Dark", "Metal", "Fairy", "Dragon"]
TYPE_COLORS = {
    "Normal": "#A8A878",
    "Fire": "#F08030",
    "Water": "#6890F0",
    "Grass": "#78C850",
    "Electric": "#F8D030",
    "Psychic": "#F85888",
    "Fighting": "#C03028",
    "Dark": "#705848",
    "Metal": "#B8B8D0",
    "Fairy": "#EE99AC",
    "Dragon": "#7038F8"
}

# Custom CSS 
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #3B5BA7;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #FFCB05;
        color: #3B5BA7;
    }
    .card-preview {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        background-color: white;
    }
    .style-preview-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
    }
    .style-preview-item {
        margin: 10px;
        text-align: center;
        cursor: pointer;
    }
    .style-preview-item img {
        border: 3px solid transparent;
        border-radius: 10px;
        transition: transform 0.3s;
    }
    .style-preview-item img:hover {
        transform: scale(1.05);
        border: 3px solid #FFCB05;
    }
    .selected-style img {
        border: 3px solid #3B5BA7 !important;
    }
    .gallery-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 16px;
    }
    .gallery-item {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .gallery-item:hover {
        transform: translateY(-5px);
    }
    .tab-content {
        padding: 20px;
        border-radius: 0 0 10px 10px;
        border: 1px solid #ddd;
        border-top: none;
    }
    /* Pokemon type badges */
    .type-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        margin-right: 5px;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# Initiate session state var 
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

if "card_style" not in st.session_state:
    st.session_state.card_style = "classic"

if "history" not in st.session_state:
    st.session_state.history = []


# Load the finetuned model 
@st.cache_resource 
def load_model():
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    lora_model_path = "duineeya/pokemon-sd-lora"

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )

        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

        if os.path.exists(lora_model_path):
            pipe.load_lora_weights(lora_model_path)
            st.success("Lora weights loaded successfully!")
        else:
            st.warning("Lora weights not found, using base model.")

        # Enable memory efficient attention if available (xformers)
        if torch.cuda.is_available():
            try:
                pipe.enable_xformers_memory_efficient_attention()
                st.success("xFormers memory efficient attention enabled")
            except:
                st.info("xFormers not available, using default attention mechanism")
                
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None 


def generate_image(prompt: str, 
                   negative_prompt: str = "", 
                   height: int = 512, 
                   width: int = 512, 
                   num_inference_steps: int = 50, 
                   guidance_scale: float = 7.5
                   ) -> Optional[Image.Image]:
    
    pipe = load_model()

    if pipe is None:
        return None 
    
    try:
        with st.spinner("Generating Pokémon image..."):
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        return image
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None