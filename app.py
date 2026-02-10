import gradio as gr
import os, subprocess, time, torch, shutil
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Carga de modelos con parche de precisión
print("🎨 Cargando SDXL Turbo...")
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device)

print("🎬 Cargando SVD XT (Parche de Precisión)...")
pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", 
    torch_dtype=torch.float16, 
    variant="fp16"
)

# --- SOLUCIÓN AL ERROR DE PRECISIÓN ---
# Forzamos el VAE a float32 para evitar el error de decodificación en T4
pipe_vid.vae.to(torch.float32)
pipe_vid.enable_sequential_cpu_offload() 

def generar_video_coherente(prompt_usuario):
    base_dir = os.getcwd()
    video_out = os.path.join(base_dir, "VCPI_Final_5s.mp4")
    
    # Director IA
    query = f"High-end cinematic prompt: {prompt_usuario}. Full body, stable anatomical limbs, realistic textures, 8k."
    prompt_ai = subprocess.check_output(["ollama", "run", "llama3", query]).decode('utf-8').strip()

    # Imagen Maestra
    image = pipe_img(prompt=prompt_ai, num_inference_steps=4, guidance_scale=1.2).images[0]
    image = image.resize((512, 512))
    image.save("master.png")

    # Generación de Video
    print("🎬 Generando 25 cuadros (5 segundos)...")
    try:
        output = pipe_vid(
            image, 
            num_frames=25, 
            motion_bucket_id=60,      # Bajamos un poco más para asegurar coherencia
            noise_aug_strength=0.04,  # Estabilidad máxima de extremidades
            decode_chunk_size=1       # Procesamiento ultra-seguro
        )
        frames = output.frames[0]
    except torch.cuda.OutOfMemoryError:
        return "Error: La GPU se quedó sin memoria. Intenta con un prompt más simple.", None

    frame_dir = "temp_frames"
    if os.path.exists(frame_dir): shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(f"{frame_dir}/{i:03d}.png")

    # Audio y Ensamblaje
    voz_path = "voz.mp3"
    subprocess.run(["edge-tts", "--text", prompt_ai[:150], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])

    subprocess.run(
        f"ffmpeg -framerate 5 -i {frame_dir}/%03d.png -i {voz_path} -c:v libx264 -pix_fmt yuv420p -shortest {video_out} -y",
        shell=True, capture_output=True
    )
    
    return prompt_ai, video_out

# Interfaz
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌌 VCPI v7.2: Producción 5s (Stable VAE)")
    with gr.Row():
        idea = gr.Textbox(label="Instrucción Visual")
        btn = gr.Button("GENERAR PRODUCCIÓN", variant="primary")
    with gr.Row():
        out_p = gr.Textbox(label="Prompt Técnico")
        out_v = gr.Video(label="Video Generado")
    
    btn.click(generar_video_coherente, inputs=[idea], outputs=[out_p, out_v])

demo.launch(share=True)
