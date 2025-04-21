from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional

# Constants
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

def wrap_text(text: str, max_width: int) -> str:
    """Wrap text to fit within max_width characters per line"""
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

    pokemon_image = pokemon_image.resize((600, 400), Image.Resampling.LANCZOS)
    card.paste(pokemon_image, (75, 150))
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
    wrapped_description = wrap_text(description, 60)
    draw.text((75, description_y), wrapped_description, fill="black", font=desc_font)
    draw.line([(75, description_y + 100), (card_width - 75, description_y + 100)], fill="gray", width=2)
    
    # Attack
    attack_y = description_y + 150
    draw.text((75, attack_y), attack_name, fill="black", font=attack_font)
    draw.text((card_width - 150, attack_y), f"{attack_damage}", fill="red", font=damage_font)
    wrapped_attack_desc = wrap_text(attack_description, 60)
    draw.text((75, attack_y + 50), wrapped_attack_desc, fill="black", font=desc_font)
    draw.rectangle(
        [(35, 35), (card_width - 35, card_height - 35)],
        outline=TYPE_COLORS.get(card_type, "#A8A878"),
        width=5
    )
    
    return card