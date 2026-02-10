import gradio as gr
import os, subprocess, time, torch
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image

# Configuración de Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carga de Modelos
print("🎨 Cargando SDXL Turbo...")
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to(device)

print("🎬 Cargando SVD XT (Versión 25 frames)...")
pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
).to(device)

# Optimizaciones de memoria indispensables para 25 frames
pipe_vid.enable_model_cpu_offload()
try:
    pipe_vid.vae.enable_slicing()
    pipe_vid.unet.enable_forward_chunking(chunk_size=1, dim=1)
except:
    pass

def pipeline_vcpi_pro(prompt_usuario):
    base_dir = os.getcwd()
    video_out = os.path.join(base_dir, "VCPI_Coherente.mp4")
    
    # 1. Ollama: Forzamos descripción anatómica completa
    try:
        query = f"Director mode: Create a high-detail prompt for: {prompt_usuario}. Mention full body, clear hands, and stable lighting. 25 words max."
        prompt_ai = subprocess.check_output(["ollama", "run", "llama3", query], timeout=30).decode('utf-8').strip()
    except:
        prompt_ai = f"Full body shot of {prompt_usuario}, highly detailed limbs, cinematic lighting, 8k."

    # 2. Imagen Maestra (Base sólida = Extremidades correctas)
    # Usamos más pasos (4) para que las manos y extremidades salgan bien definidas
    image = pipe_img(prompt=prompt_ai, num_inference_steps=4, guidance_scale=1.0).images[0]
    image = image.resize((512, 512))
    image.save("base.png")

    # 3. Animación de 5 segundos (25 frames a 5 fps o interpolación)
    # Aumentamos motion_bucket_id para que haya movimiento pero con coherencia
    print("🎬 Generando 25 cuadros de alta coherencia...")
    frames = pipe_vid(
        image, 
        decode_chunk_size=2, 
        num_frames=25, # Máximo del modelo para duración
        motion_bucket_id=100, # Menos distorsión en extremidades
        noise_aug_strength=0.1 # Más fidelidad a la imagen original
    ).frames[0]
    
    frame_dir = "temp_frames"
    if os.path.exists(frame_dir): shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(f"{frame_dir}/{i:03d}.png")

    # 4. Voz
    voz_path = "voz.mp3"
    subprocess.run(["edge-tts", "--text", prompt_ai[:150], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])

    # 5. FFMPEG: Ajustamos a 5 fps para dar 5 segundos de video real
    # O podemos usar 'minterpolate' para suavizar a 24fps
    print("🎥 Ensamblando video de larga duración...")
    subprocess.run(
        f"ffmpeg -framerate 5 -i {frame_dir}/%03d.png -i {voz_path} -c:v libx264 -pix_fmt yuv420p -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -shortest {video_out} -y",
        shell=True, capture_output=True
    )
    
    return prompt_ai, video_out

# Interfaz
with gr.Blocks() as demo:
    gr.Markdown("# 🌌 VCPI v7.0 - Coherencia Anatómica y 5 Segundos")
    with gr.Row():
        in_t = gr.Textbox(label="Describe la escena (Ej: Científico trabajando de cuerpo completo)")
        btn = gr.Button("GENERAR VIDEO LARGO", variant="primary")
    with gr.Row():
        out_p = gr.Textbox(label="Prompt Técnico")
        out_v = gr.Video(label="Resultado 5s")
    
    btn.click(pipeline_vcpi_pro, inputs=[in_t], outputs=[out_p, out_v])

demo.launch(share=True)
