import gradio as gr
import os, subprocess, time, torch, shutil
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image

# Configuración Dual-GPU Balanceada
device_primary = "cuda:0"   # Ayuda con VAE y genera imágenes
device_secondary = "cuda:1" # Proceso principal de SVD (Unet)

# 1. Carga de modelos distribuida
# Ponemos SDXL en la GPU 0
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to(device_primary)

# Cargamos SVD en la GPU 1, pero...
pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device_secondary)

# --- BALANCEO MANUAL DE CARGA ---
# Movemos el VAE (que decodifica los frames) a la GPU 0. 
# Esto quita peso a la GPU 1 cuando llega al final de la animación.
pipe_vid.vae.to(device_primary) 
pipe_vid.vae.to(torch.float32) # Estabilidad anatómica

# Optimizaciones de memoria para GPU 1
pipe_vid.enable_sequential_cpu_offload(device=device_secondary)

def produccion_balanceada(prompt_usuario):
    base_dir = "/kaggle/working/VCPI"
    video_out = os.path.join(base_dir, "VCPI_Final_5s.mp4")
    frame_dir = os.path.join(base_dir, "temp_frames")
    
    if os.path.exists(frame_dir): shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    # 1. Ollama (Cerebro)
    query = f"Director: Expand this for a 5s cinematic clip with perfect human anatomy: {prompt_usuario}."
    prompt_ai = subprocess.check_output(["ollama", "run", "llama3", query]).decode('utf-8').strip()

    # 2. GPU 0: Generar Imagen Maestra
    image = pipe_img(prompt=prompt_ai, num_inference_steps=4, guidance_scale=1.2).images[0]
    image = image.resize((512, 512))
    image.save("master.png")

    # 3. Animación (GPU 1 hace el UNET, GPU 0 hace el VAE)
    print("🎬 [GPU 1 + GPU 0] Trabajando en equipo para 25 cuadros...")
    try:
        # El pipeline detectará automáticamente que el VAE está en la otra GPU
        output = pipe_vid(
            image, 
            num_frames=25, 
            motion_bucket_id=70, 
            noise_aug_strength=0.04,
            decode_chunk_size=1 
        )
        frames = output.frames[0]
    except Exception as e:
        return f"Error en procesamiento dual: {e}", None

    # 4. Guardado y FFMPEG
    for i, frame in enumerate(frames):
        frame.save(f"{frame_dir}/{i:03d}.png")

    voz_path = os.path.join(base_dir, "voz.mp3")
    subprocess.run(["edge-tts", "--text", prompt_ai[:150], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])
    
    # FFMPEG optimizado
    subprocess.run(
        f"ffmpeg -framerate 5 -i {frame_dir}/%03d.png -i {voz_path} -c:v libx264 -pix_fmt yuv420p -movflags +faststart {video_out} -y",
        shell=True, capture_output=True
    )
    
    return prompt_ai, video_out

# Interfaz
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🌌 VCPI v9.0 - Dual GPU Hybrid Load Balance")
    with gr.Row():
        input_t = gr.Textbox(label="Instrucción Visual")
        btn = gr.Button("GENERAR PRODUCCIÓN HÍBRIDA", variant="primary")
    with gr.Row():
        output_t = gr.Textbox(label="Prompt Final")
        output_v = gr.Video(label="Video Generado")

    btn.click(produccion_balanceada, inputs=[input_t], outputs=[output_t, output_v])

demo.launch(share=True)
