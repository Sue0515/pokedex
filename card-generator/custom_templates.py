import os
import io
import base64
import torch

from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Tuple, Optional, List


CARD_STYLE_TEMPLATES = {
    "classic": {
        "name": "Classic",
        "description": "Original Pokémon card style with simple layout",
        "card_width": 750,
        "card_height": 1050,
        "image_position": (75, 150),
        "image_size": (600, 400),
        "border_width": 5,
        "inner_margin": 40,
        "name_position": (75, 80),
        "hp_position": (630, 80),
        "type_badge_position": (570, 140),
        "type_badge_radius": 25,
        "description_position": (75, 580),
        "attack_name_position": (75, 730),
        "attack_damage_position": (600, 730),
        "attack_description_position": (75, 780),
        "separator_line": [(75, 680), (675, 680)]
    },
    "modern": {
        "name": "Modern",
        "description": "Modern style with sleek design and rounded corners",
        "card_width": 750,
        "card_height": 1050,
        "image_position": (75, 150),
        "image_size": (600, 425),
        "border_width": 8,
        "inner_margin": 35,
        "name_position": (75, 75),
        "hp_position": (650, 75),
        "type_badge_position": (580, 135),
        "type_badge_radius": 30,
        "description_position": (75, 600),
        "attack_name_position": (75, 750),
        "attack_damage_position": (620, 750),
        "attack_description_position": (75, 800),
        "separator_line": [(75, 700), (675, 700)],
        "rounded_corners": 30,
        "shadow_offset": 8
    },
    "vintage": {
        "name": "Vintage",
        "description": "Retro style with worn edges and aged look",
        "card_width": 750,
        "card_height": 1050,
        "image_position": (85, 160),
        "image_size": (580, 380),
        "border_width": 10,
        "inner_margin": 50,
        "name_position": (85, 90),
        "hp_position": (610, 90),
        "type_badge_position": (550, 150),
        "type_badge_radius": 25,
        "description_position": (85, 570),
        "attack_name_position": (85, 720),
        "attack_damage_position": (590, 720),
        "attack_description_position": (85, 770),
        "separator_line": [(85, 670), (665, 670)],
        "texture_overlay": True
    },
    "holographic": {
        "name": "Holographic",
        "description": "Shiny holographic style with rainbow effects",
        "card_width": 750,
        "card_height": 1050,
        "image_position": (75, 150),
        "image_size": (600, 400),
        "border_width": 12,
        "inner_margin": 45,
        "name_position": (75, 80),
        "hp_position": (630, 80),
        "type_badge_position": (570, 140),
        "type_badge_radius": 28,
        "description_position": (75, 580),
        "attack_name_position": (75, 730),
        "attack_damage_position": (600, 730),
        "attack_description_position": (75, 780),
        "separator_line": [(75, 680), (675, 680)],
        "holographic_effect": True
    }
}

# Type-specific gradients for modern style
TYPE_GRADIENTS = {
    "Normal": ["#A8A878", "#CACA98"],
    "Fire": ["#F08030", "#FA9E50"],
    "Water": ["#6890F0", "#88B0FF"],
    "Grass": ["#78C850", "#98E870"],
    "Electric": ["#F8D030", "#FFEE70"],
    "Psychic": ["#F85888", "#FFA8B8"],
    "Fighting": ["#C03028", "#E05048"],
    "Dark": ["#705848", "#907868"],
    "Metal": ["#B8B8D0", "#D8D8F0"],
    "Fairy": ["#EE99AC", "#FFBCCC"],
    "Dragon": ["#7038F8", "#9058FF"]
}
