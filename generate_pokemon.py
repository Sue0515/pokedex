import os 
import datetime
import torch
import argparse
from diffusers import StableDiffusionPipeline 

# CUDA diagnostics
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name()}")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate Pokémon images using fine-tuned Stable Diffusion")
parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
parser.add_argument("--negative_prompt", type=str, default="low quality, bad anatomy, worst quality, low resolution", 
                    help="Negative prompt to avoid unwanted elements")
parser.add_argument("--output_dir", type=str, default="generated_images", 
                    help="Directory to save generated images")
parser.add_argument("--steps", type=int, default=50, 
                    help="Number of inference steps")
parser.add_argument("--guidance_scale", type=float, default=7.5, 
                    help="Guidance scale for prompt adherence")

args = parser.parse_args()

# Create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

lora_weights_path = "duineeya/pokemon-sd-lora"
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

pipe = pipe.to(device)

try:
    pipe.enable_attention_slicing()
    print("Attention slicing enabled")
except Exception as e:
    print(f"Could not enable attention slicing: {e}")

# Try to load LoRA weights with error handling
try:
    pipe.load_lora_weights(lora_weights_path)
    print("LoRA weights loaded successfully")
except ValueError as e:
    if "PEFT backend is required" in str(e):
        print("Error: PEFT backend is required. Please install it with 'pip install peft'")
        exit(1)
    else:
        print(f"Error loading LoRA weights: {e}")
        exit(1)


print("Generating Image...")
# Generate an image using command line arguments
image = pipe(
    prompt=args.prompt,
    negative_prompt=args.negative_prompt,
    num_inference_steps=args.steps,
    guidance_scale=args.guidance_scale,
    height=512,
    width=512
).images[0]

current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a filename based on a shortened version of the prompt
short_prompt = args.prompt[:30].replace(" ", "_").replace(",", "").replace(".", "")
fname = f"{short_prompt}_{current_datetime}.png"
full_path = os.path.join(args.output_dir, fname)

# Save the image to the specified path
image.save(full_path)
print(f"Generated image for prompt: \"{args.prompt}\"")
print(f"Image saved to: {full_path}")