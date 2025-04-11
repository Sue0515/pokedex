import os 
import datetime
import torch 
from diffusers import StableDiffusionPipeline 

output_dir = "./generated_results"
lora_weights_path = "duineeya/pokemon-sd-lora"
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention() 


pipe.load_lora_weights(lora_weights_path)

# Generate an image
prompt = "A cute blue electric mouse Pokémon with yellow cheeks"
negative_prompt = "low quality, bad anatomy, worst quality, low resolution"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]

current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

fname = f"pokemon_image_{current_datetime}.png"
full_path = os.path.join(output_dir, fname)
image.save(full_path)
