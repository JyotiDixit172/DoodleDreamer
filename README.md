Project Overview:
Draw2Life is an AI-powered application that transforms hand-drawn sketches into realistic images. 
It integrates two powerful models—BLIP for image captioning and Stable Diffusion for image generation.
The process begins with BLIP generating a descriptive caption from a doodle.
This caption is then refined to remove sketch-related phrases and converted into a prompt.
The refined prompt is passed to Stable Diffusion, which generates a high-resolution, photorealistic image.
The entire workflow is wrapped in a user-friendly Gradio interface, allowing users to upload sketches and instantly view both the generated caption and the realistic image.
The project uses the Kaggle Doodle Dataset as input and explores the creative potential of generative AI in visual interpretation.

To run the app run the following commands:
1. pip install torch torchvision pillow transformers diffusers matplotlib gradio
2. python app.py
