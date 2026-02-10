import gradio as gr
import os, subprocess, torch
import torch
from diffusers import StableVideoDiffusionPipeline, StableDiffusionPipeline
from PIL import Image
import time

# Configuración de Modelos (Nivel Sora/Veo)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargamos el generador de imágenes (SDXL o SD 1.5)
pipe_img = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)

# Cargamos el generador de video (Stable Video Diffusion)
pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "compatibility/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
).to(device)

def pipeline_pro(prompt, duracion):
    base_dir = os.getcwd()
    video_out = os.path.join(base_dir, "IA_Video_Cinematic.mp4")
    
    # 1. Ollama: Guion y Prompt Mejorado
    ollama_bin = os.getenv('OLLAMA_BIN', 'ollama')
    guion = subprocess.check_output([ollama_bin, "run", "llama3", f"Escribe un prompt descriptivo para una IA de video sobre: {prompt}. Sé muy visual."], timeout=30).decode('utf-8')

    # 2. Generación de Imagen Base (La "foto" inicial)
    print("🎨 Generando imagen base realista...")
    image = pipe_img(prompt + ", photorealistic, 8k, highly detailed, cinematic lighting").images[0]
    image = image.resize((512, 512))
    
    # 3. Generación de Video (Darle vida a la imagen)
    print("🎬 Generando movimiento nivel Veo...")
    frames = pipe_vid(image, decode_chunk_size=8).frames[0]
    
    # Guardar frames como video usando ffmpeg
    temp_folder = "temp_frames"
    os.makedirs(temp_folder, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(f"{temp_folder}/frame_{i:04d}.png")

    # 4. Voz (TTS)
    voz_path = os.path.join(base_dir, "voz.mp3")
    subprocess.run(["edge-tts", "--text", guion[:100], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])

    # 5. Ensamblaje Final con Audio
    subprocess.run(f"ffmpeg -framerate 7 -i {temp_folder}/frame_%04d.png -i {voz_path} -c:v libx264 -pix_fmt yuv420p -shortest {video_out} -y", shell=True)

    return guion, video_out

# Interfaz Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌌 VCPI Pro: Sora-Level Video Gen")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Instrucción Visual")
            btn = gr.Button("GENERAR VIDEO IA", variant="primary")
        with gr.Column():
            out_v = gr.Video()
            out_t = gr.Textbox(label="Descripción de la Escena")

    btn.click(pipeline_pro, [idea, gr.State(10)], [out_t, out_v])

demo.launch(share=True)
