import gradio as gr
import os, subprocess, torch, time
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image
from PIL import Image

# Cargar modelos en la GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pipeline de Imagen (SDXL Turbo - Ultra rápido para rostros/escenarios)
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to(device)

# Pipeline de Video (SVD - Para animar la imagen con realismo)
pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
).to(device)
pipe_vid.enable_model_cpu_offload() # Optimización de VRAM

def generar_produccion(prompt_usuario):
    base_dir = "/kaggle/working/VCPI"
    video_out = os.path.join(base_dir, "VCPI_AI_SORA.mp4")
    
    # 1. Ollama: Mejora el prompt (Autonomía)
    # Convierte "un rostro" en "Primer plano de un androide con piel de silicona, ojos ámbar..."
    prompt_ai = subprocess.check_output(
        ["ollama", "run", "llama3", f"Expande este prompt a una escena cinematográfica de 20 palabras: {prompt_usuario}"],
        timeout=30
    ).decode('utf-8').strip()

    # 2. Generar Imagen Maestra
    print("🎨 Creando visuales de alta fidelidad...")
    imagen = pipe_img(prompt=prompt_ai, num_inference_steps=2, guidance_scale=0.0).images[0]
    imagen = imagen.resize((512, 512))
    img_path = os.path.join(base_dir, "base_ia.png")
    imagen.save(img_path)

    # 3. Animar Imagen (Generación de Video)
    print("🎬 Animando escena...")
    frames = pipe_vid(imagen, decode_chunk_size=8, motion_bucket_id=127).frames[0]
    
    # Guardar frames para FFMPEG
    frame_dir = os.path.join(base_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(f"{frame_dir}/{i:03d}.png")

    # 4. Voz y Audio
    voz_path = os.path.join(base_dir, "voz.mp3")
    subprocess.run(["edge-tts", "--text", prompt_ai[:150], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])

    # 5. Ensamblaje Final
    subprocess.run(
        f"ffmpeg -framerate 7 -i {frame_dir}/%03d.png -i {voz_path} -c:v libx264 -pix_fmt yuv420p -shortest {video_out} -y",
        shell=True
    )
    
    return prompt_ai, video_out

# Interfaz Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌌 VCPI - Generación Autónoma nivel Sora")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Prompt (Rostros, Escenarios, Ciencia Ficción)")
            btn = gr.Button("GENERAR VIDEO IA", variant="primary")
        with gr.Column():
            out_v = gr.Video()
            out_t = gr.Textbox(label="Interpretación de la IA")

    btn.click(generar_produccion, [idea], [out_t, out_v])

demo.launch(share=True)
