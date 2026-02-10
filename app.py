import gradio as gr
import os, subprocess, torch, time
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image
from PIL import Image

# Cargar Motores de IA (Optimizados para T4)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Motor de Imagen (Para rostros y escenarios detallados)
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to(device)

# Motor de Video (Para el movimiento fluido)
pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
).to(device)
pipe_vid.enable_model_cpu_offload() # Ahorro de VRAM

def generar_autonomo(prompt_usuario):
    base_dir = os.getcwd()
    video_out = os.path.join(base_dir, "VCPI_PRO_AI.mp4")
    
    # 1. DIRECTOR (Ollama): Mejora el prompt para hacerlo cinematográfico
    director_cmd = f"Mejora este prompt para una IA de video, añade detalles de iluminación y cámara: {prompt_usuario}"
    prompt_ai = subprocess.check_output(["ollama", "run", "llama3", director_cmd]).decode('utf-8').strip()[:200]
    
    # 2. ESCENÓGRAFO (SDXL): Crea la imagen base (Rostros, Laboratorios, etc.)
    print("🎨 Creando escenario base...")
    imagen_base = pipe_img(prompt=prompt_ai, num_inference_steps=2, guidance_scale=0.0).images[0]
    imagen_base = imagen_base.resize((512, 512))
    imagen_base_path = os.path.join(base_dir, "base.png")
    imagen_base.save(imagen_base_path)

    # 3. ANIMADOR (SVD): Genera el video con IA
    print("🎬 Animando escena (Nivel Sora)...")
    # Generamos 14-25 frames con coherencia temporal
    frames = pipe_vid(imagen_base, decode_chunk_size=8, motion_bucket_id=127, fps=7).frames[0]
    
    # Guardar frames temporales
    frame_dir = os.path.join(base_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(f"{frame_dir}/{i:03d}.png")

    # 4. LOCUCIÓN (Edge-TTS)
    voz_path = os.path.join(base_dir, "voz.mp3")
    subprocess.run(["edge-tts", "--text", prompt_ai[:150], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])

    # 5. ENSAMBLAJE (FFMPEG)
    subprocess.run(
        f"ffmpeg -framerate 7 -i {frame_dir}/%03d.png -i {voz_path} -c:v libx264 -pix_fmt yuv420p -shortest {video_out} -y", 
        shell=True
    )
    
    return prompt_ai, video_out

# Interfaz Gradio Pro
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🌌 VCPI - Sistema Autónomo de Video IA")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Instrucción (Ej: Primer plano de un anciano en el año 3000)")
            btn = gr.Button("GENERAR PRODUCCIÓN COMPLETA", variant="primary")
        with gr.Column():
            res_v = gr.Video(label="Video Generado por IA")
            res_t = gr.Textbox(label="Interpretación de la IA")

    btn.click(generar_autonomo, inputs=[idea], outputs=[res_t, res_v])

demo.launch(share=True)
