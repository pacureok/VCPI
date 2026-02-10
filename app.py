import gradio as gr
import os, subprocess, time, torch
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image

# 1. Configuración de Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Carga de Modelos
print("🎨 Cargando Generador de Imágenes (SDXL)...")
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to(device)

print("🎬 Cargando Generador de Video (SVD)...")
pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
).to(device)

# --- OPTIMIZACIONES DE MEMORIA (Solución al NotImplementedError) ---
pipe_vid.enable_model_cpu_offload()

# Intentamos slicing, que es compatible con el decodificador temporal
try:
    pipe_vid.vae.enable_slicing()
    pipe_vid.unet.enable_forward_chunking(chunk_size=1, dim=1)
    print("✅ Optimizaciones de memoria activadas.")
except Exception as e:
    print(f"⚠️ Aviso: Algunas optimizaciones no se aplicaron: {e}")

def pipeline_vcpi(prompt_usuario):
    base_dir = os.getcwd()
    video_out = os.path.join(base_dir, "VCPI_Final.mp4")
    
    # 1. Autonomía con Ollama
    try:
        query = f"Act as a film director. Create a high-end cinematic prompt for: {prompt_usuario}. 20 words max."
        prompt_ai = subprocess.check_output(["ollama", "run", "llama3", query], timeout=30).decode('utf-8').strip()
    except:
        prompt_ai = f"Cinematic shot of {prompt_usuario}, 8k, realistic."

    # 2. Imagen Maestra
    image = pipe_img(prompt=prompt_ai, num_inference_steps=2, guidance_scale=0.0).images[0]
    image = image.resize((512, 512))
    image.save("base.png")

    # 3. Animación (14 frames para estabilidad)
    frames = pipe_vid(image, decode_chunk_size=2, num_frames=14, motion_bucket_id=127).frames[0]
    
    frame_dir = "temp_frames"
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(f"{frame_dir}/{i:03d}.png")

    # 4. Voz
    voz_path = "voz.mp3"
    subprocess.run(["edge-tts", "--text", prompt_ai[:150], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])

    # 5. FFMPEG
    subprocess.run(
        f"ffmpeg -framerate 7 -i {frame_dir}/%03d.png -i {voz_path} -c:v libx264 -pix_fmt yuv420p -shortest {video_out} -y",
        shell=True, capture_output=True
    )
    
    return prompt_ai, video_out

# Interfaz
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🌌 VCPI v6.0 - High End AI Video")
    with gr.Row():
        in_t = gr.Textbox(label="Tu visión (Rostros, escenarios, etc.)")
        btn = gr.Button("GENERAR PRODUCCIÓN", variant="primary")
    with gr.Row():
        out_p = gr.Textbox(label="Prompt del Director")
        out_v = gr.Video(label="Video Final")
    
    btn.click(pipeline_vcpi, inputs=[in_t], outputs=[out_p, out_v])

demo.launch(share=True)
