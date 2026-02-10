import gradio as gr
import os, subprocess, time, torch, shutil
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image

# Configuración de Dispositivos
device_0 = "cuda:0" # GPU de apoyo (Imagen + Decodificación VAE)
device_1 = "cuda:1" # GPU principal (Cálculo de movimiento SVD)

# 1. Carga de modelos con Balanceo Híbrido
print("🎨 Cargando SDXL en GPU 0...")
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to(device_0)

print("🎬 Cargando SVD en GPU 1...")
pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device_1)

# --- TRUCO DE BALANCEO ---
# Movemos el VAE (lo que convierte números en imágenes) a la GPU 0.
# Esto hace que la GPU 0 trabaje mientras la GPU 1 calcula el video.
pipe_vid.vae.to(device_0)
pipe_vid.vae.to(torch.float32) # Mayor calidad y estabilidad

# Optimizaciones de memoria para GPU 1
pipe_vid.enable_sequential_cpu_offload(device=device_1)

def produccion_hibrida(prompt_usuario):
    base_dir = "/kaggle/working/VCPI"
    video_out = os.path.join(base_dir, "output_5s.mp4")
    frame_dir = os.path.join(base_dir, "temp_frames")
    
    if os.path.exists(frame_dir): shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    # 1. Ollama: Prompt Cinematográfico
    query = f"Director: Create a 5s cinematic prompt for: {prompt_usuario}. Ensure full anatomical coherence and stable limbs."
    prompt_ai = subprocess.check_output(["ollama", "run", "llama3", query]).decode('utf-8').strip()

    # 2. Generar Imagen Base (GPU 0)
    print(f"🖼️ [GPU 0] Generando base...")
    image = pipe_img(prompt=prompt_ai, num_inference_steps=4, guidance_scale=1.2).images[0]
    image = image.resize((512, 512))
    image.save("master.png")

    # 3. Generar Video (GPU 1 Unet + GPU 0 VAE)
    print(f"🎬 [GPU 1+0] Generando 25 cuadros coherentes...")
    output = pipe_vid(
        image, 
        num_frames=25, 
        motion_bucket_id=70, 
        noise_aug_strength=0.04,
        decode_chunk_size=1 
    )
    frames = output.frames[0]

    # 4. Guardado de cuadros
    for i, frame in enumerate(frames):
        frame.save(f"{frame_dir}/{i:03d}.png")

    # 5. Audio y Ensamblaje (FFMPEG)
    voz_path = os.path.join(base_dir, "voz.mp3")
    subprocess.run(["edge-tts", "--text", prompt_ai[:150], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])
    
    # 5fps para que 25 cuadros duren 5 segundos
    subprocess.run(
        f"ffmpeg -framerate 5 -i {frame_dir}/%03d.png -i {voz_path} -c:v libx264 -pix_fmt yuv420p -movflags +faststart {video_out} -y",
        shell=True, capture_output=True
    )
    
    return prompt_ai, video_out

# Interfaz Gradio
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🌌 VCPI v9.0 - Dual GPU Hybrid System")
    gr.Markdown("Balanceo de carga: GPU 1 (90% MoCap) | GPU 0 (50% Rendering)")
    
    with gr.Row():
        idea = gr.Textbox(label="Instrucción de Video")
        btn = gr.Button("GENERAR PRODUCCIÓN 5S", variant="primary")
    with gr.Row():
        out_t = gr.Textbox(label="Prompt de la IA")
        out_v = gr.Video(label="Resultado Final")

    btn.click(produccion_hibrida, inputs=[idea], outputs=[out_t, out_v])

demo.launch(share=True)
