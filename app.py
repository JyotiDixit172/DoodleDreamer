# 📁 Step 2: Import libraries
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import gradio as gr
# 🔌 Step 3: Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# 🧠 Step 4: Load BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
# 🎨 Step 5: Load Stable Diffusion Pipeline
pipe = StableDiffusionPipeline.from_pretrained(
   "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
).to(device)

# 🎛️ Step 6: Define function for Gradio UI
def sketch_to_realistic(image):
   image = image.convert("RGB")
   inputs = blip_processor(images=image, return_tensors="pt").to(device)
   caption_ids = blip_model.generate(**inputs)
   caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)

   refined_caption = caption.lower().replace("how to draw a", "").strip()
   refined_caption = caption.lower().replace("drawing of a", "").strip()
   prompt = f"A high-resolution realistic photo of {refined_caption}"

   result = pipe(prompt).images[0]
   return caption, result

# 🌐 Step 7: Build Gradio interface
demo = gr.Interface(
   fn=sketch_to_realistic,
   inputs=gr.Image(type="pil", label="Upload Doodle (Sketch)"),
   outputs=[
       gr.Textbox(label="Generated Caption"),
       gr.Image(label="Generated Realistic Image")
   ],
   title="🎨 Draw2Life : Sketch to Realistic Image Generator",
   description="Upload a doodle/sketch. The system will caption it using BLIP and generate a realistic image using Stable Diffusion."
)
# 🚀 Step 8: Launch Gradio
demo.launch(debug=True)