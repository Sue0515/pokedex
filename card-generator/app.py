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

os.environ['STREAMLIT_SERVER_PORT'] = '8501'
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

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


def wrap_text(text: str, max_width: int) -> str:
    words = text.split() 
    lines = []
    current_line = []

    for word in words:
        if len(" ".join(current_line + [word])) <= max_width:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)


def create_pokemon_card(
        pokemon_image: Image.Image,
        name: str,
        hp: int, 
        card_type: str,
        description: str,
        attack_name: str, 
        attack_damage: int,
        attack_description: str
) -> Image.Image:
    """Create a Pokémon card with the generated image and given details"""

    card_width = 750
    card_height = 1050

    # Create a blank card with the type's color
    card = Image.new('RGB', (card_width, card_height), TYPE_COLORS.get(card_type, "#A8A878"))
    draw = ImageDraw.Draw(card)


    # Create inner white rectangle
    draw.rectangle(
        [(40, 40), (card_width - 40, card_height - 40)],
        fill="white"
    )

    # Try to load fonts (use default if not available)
    try:
        try:
            name_font = ImageFont.truetype("Arial Bold.ttf", 48)
        except:
            try:
                name_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
            except:
                name_font = ImageFont.load_default()
                
        try:
            hp_font = ImageFont.truetype("Arial Bold.ttf", 40)
        except:
            try:
                hp_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
            except:
                hp_font = ImageFont.load_default()
                
        try:
            type_font = ImageFont.truetype("Arial.ttf", 30)
        except:
            try:
                type_font = ImageFont.truetype("DejaVuSans.ttf", 30)
            except:
                type_font = ImageFont.load_default()
                
        try:
            desc_font = ImageFont.truetype("Arial.ttf", 24)
        except:
            try:
                desc_font = ImageFont.truetype("DejaVuSans.ttf", 24)
            except:
                desc_font = ImageFont.load_default()
                
        try:
            attack_font = ImageFont.truetype("Arial Bold.ttf", 36)
        except:
            try:
                attack_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
            except:
                attack_font = ImageFont.load_default()
                
        try:
            damage_font = ImageFont.truetype("Arial Bold.ttf", 42)
        except:
            try:
                damage_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
            except:
                damage_font = ImageFont.load_default()
    except IOError:
        name_font = ImageFont.load_default()
        hp_font = ImageFont.load_default()
        type_font = ImageFont.load_default()
        desc_font = ImageFont.load_default()
        attack_font = ImageFont.load_default()
        damage_font = ImageFont.load_default()

    # Resize and paste the Pokemon image 
    pokemon_image = pokemon_image.resize((600, 400), Image.Resampling.LANCZOS)
    card.paste(pokemon_image, (75, 150))

    # Add card details
    draw.text((75, 80), name, fill="black", font=name_font)
    draw.text((card_width - 120, 80), f"HP {hp}", fill="red", font=hp_font)
    
    # Type badge
    type_badge_x = card_width - 180
    type_badge_y = 140
    draw.ellipse(
        [(type_badge_x - 25, type_badge_y - 25), (type_badge_x + 25, type_badge_y + 25)],
        fill=TYPE_COLORS.get(card_type, "#A8A878")
    )
    draw.text((type_badge_x - 15, type_badge_y - 15), card_type[0], fill="white", font=type_font)
    
    # Description
    description_y = 580
    # Wrap text to fit the card width
    wrapped_description = wrap_text(description, 60)
    draw.text((75, description_y), wrapped_description, fill="black", font=desc_font)
    
    # Draw a line to separate description from attack
    draw.line([(75, description_y + 100), (card_width - 75, description_y + 100)], fill="gray", width=2)
    
    # Attack
    attack_y = description_y + 150
    draw.text((75, attack_y), attack_name, fill="black", font=attack_font)
    draw.text((card_width - 150, attack_y), f"{attack_damage}", fill="red", font=damage_font)
    
    # Wrap attack description
    wrapped_attack_desc = wrap_text(attack_description, 60)
    draw.text((75, attack_y + 50), wrapped_attack_desc, fill="black", font=desc_font)
    
    # Add a subtle frame border
    draw.rectangle(
        [(35, 35), (card_width - 35, card_height - 35)],
        outline=TYPE_COLORS.get(card_type, "#A8A878"),
        width=5
    )
    
    return card


def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def main():
    st.title("Pokémon Card Generator 🎴")
    st.write("Create your own custom Pokémon card using a fine-tuned Stable Diffusion model")
    
    with st.sidebar:
        st.header("Card Details")
        pokemon_name = st.text_input("Pokémon Name", "Pikablu")
        pokemon_hp = st.slider("HP", 10, 300, 100, 10)
        pokemon_type = st.selectbox("Type", CARD_TYPES)
        pokemon_description = st.text_area("Description", "A rare electric water type Pokémon found in the mountains.")
        
        st.header("Attack Details")
        attack_name = st.text_input("Attack Name", "Thunder Splash")
        attack_damage = st.slider("Attack Damage", 10, 250, 70, 10)
        attack_description = st.text_area("Attack Description", "Flip a coin. If heads, the defending Pokémon is now paralyzed.")
    
    st.header("Image Generation")
    prompt = st.text_area(
        "Describe the Pokémon you want to generate",
        "A cute blue electric mouse Pokémon with yellow cheeks, digital art style"
    )
    
    negative_prompt = st.text_area(
        "Negative Prompt (what to avoid in the image)",
        "low quality, bad anatomy, worst quality, low resolution, deformed features, text, blurry"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_steps = st.slider("Number of Inference Steps", 20, 100, 50)
    
    with col2:
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
    
    with col3:
        image_size = st.selectbox("Image Size", ["Square (512x512)", "Portrait (512x768)", "Landscape (768x512)"])
        
        if image_size == "Square (512x512)":
            height, width = 512, 512
        elif image_size == "Portrait (512x768)":
            height, width = 768, 512
        else:  # Landscape
            height, width = 512, 768
    
    # Add examples of prompts that work well with the model
    with st.expander("📚 Prompt Examples"):
        st.markdown("""
        ### Try these prompts for good results:
        
        - "A cute dragon Pokémon with scales and tiny wings, detailed digital art"
        - "A plantlike Pokémon with flower petals and vines, anime style"
        - "A crystal rock Pokémon with geometric shapes, glowing elements, high quality"
        - "A fluffy cloud Pokémon with a smiling face, pastel colors, kawaii style"
        - "A mechanical Pokémon with gears and metallic parts, steampunk style"
        
        **Pro tip:** Be specific about colors, features, and art style to get the best results.
        """)

    if st.button("Generate", key="generate"):
        pokemon_image = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale
        )

        if pokemon_image:
            # Create the card
            card = create_pokemon_card(
                pokemon_image=pokemon_image,
                name=pokemon_name,
                hp=pokemon_hp,
                card_type=pokemon_type,
                description=pokemon_description,
                attack_name=attack_name,
                attack_damage=attack_damage,
                attack_description=attack_description
            )
            
            # Display the card
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Generated Pokémon")
                st.image(pokemon_image, use_column_width=True)
                st.markdown(
                    get_image_download_link(pokemon_image, f"{pokemon_name.lower().replace(' ', '_')}.png", "Download Pokémon Image"),
                    unsafe_allow_html=True
                )
            
            with col2:
                st.subheader("Final Pokémon Card")
                st.image(card, use_column_width=True)
                st.markdown(
                    get_image_download_link(card, f"{pokemon_name.lower().replace(' ', '_')}_card.png", "Download Pokémon Card"),
                    unsafe_allow_html=True
                )

    st.markdown("---")
    st.markdown("### About this app")
    st.write("This app uses a fine-tuned Stable Diffusion v1.5 model to generate Pokémon images based on your descriptions. The model was fine-tuned using Low-Rank Adaptation (LoRA) technique.")


if __name__ == "__main__":
    main() 