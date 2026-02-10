import gradio as gr
import os, subprocess, time, torch
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image
from PIL import Image

# 1. Configuración de Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Carga de Modelos Generativos
print("⏳ Cargando motores de IA (esto puede tardar un poco en la primera ejecución)...")

# Motor de Imagen: SDXL Turbo (Para rostros y escenarios ultra-detallados)
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device)

# Motor de Video: SVD XT (Para movimiento fluido nivel Sora)
pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device)

# --- OPTIMIZACIONES ANTI-CRASH (MEMORIA T4) ---
pipe_vid.enable_model_cpu_offload()
pipe_vid.unet.enable_forward_chunking(chunk_size=1, dim=1)

# Parche para el error de 'AutoencoderKLTemporalDecoder'
if hasattr(pipe_vid.vae, 'enable_tiling'):
    pipe_vid.vae.enable_tiling()
else:
    pipe_vid.vae.enable_slicing() 

def pipeline_maestro(prompt_usuario):
    base_dir = os.getcwd()
    video_out = os.path.join(base_dir, "VCPI_Final_AI.mp4")
    
    # 1. DIRECTOR IA (Ollama): Autonomía creativa
    try:
        # Pedimos a Llama 3 que actúe como director técnico de fotografía
        instruccion = f"Transforma este concepto en un prompt cinematográfico de 20 palabras para video (en inglés): '{prompt_usuario}'. Incluye detalles de luz, texturas y atmósfera."
        prompt_ai = subprocess.check_output(["ollama", "run", "llama3", instruccion], timeout=30).decode('utf-8').strip()
    except:
        prompt_ai = f"Cinematic close-up of {prompt_usuario}, high detail, 8k, realistic textures."

    # 2. GENERACIÓN DE IMAGEN (SDXL)
    print(f"🎨 Creando imagen base: {prompt_ai[:50]}...")
    image = pipe_img(prompt=prompt_ai, num_inference_steps=2, guidance_scale=0.0).images[0]
    image = image.resize((512, 512))
    img_path = os.path.join(base_dir, "frame_maestro.png")
    image.save(img_path)

    # 3. GENERACIÓN DE VIDEO (SVD)
    print("🎬 Animando escena (Nivel Sora)...")
    try:
        # Generamos 14 frames para máxima estabilidad en Kaggle
        frames = pipe_vid(
            image, 
            decode_chunk_size=2, 
            motion_bucket_id=127, 
            num_frames=14, 
            fps=7
        ).frames[0]
        
        # Carpeta temporal para frames
        temp_frames = os.path.join(base_dir, "temp_frames")
        os.makedirs(temp_frames, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(f"{temp_frames}/{i:03d}.png")
            
    except Exception as e:
        return f"Error en animación: {e}", None

    # 4. LOCUCIÓN (Edge-TTS)
    voz_path = os.path.join(base_dir, "voz.mp3")
    subprocess.run(["edge-tts", "--text", prompt_ai[:150], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])

    # 5. ENSAMBLAJE FINAL (FFMPEG)
    print("🎥 Ensamblando video final...")
    subprocess.run(
        f"ffmpeg -framerate 7 -i {temp_frames}/%03d.png -i {voz_path} -c:v libx264 -pix_fmt yuv420p -shortest {video_out} -y",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    return prompt_ai, video_out

# --- INTERFAZ GRADIO ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🌌 VCPI - Sistema de Video Autónomo IA")
    gr.Markdown("Generación de rostros, escenarios y entornos fotorrealistas.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Describe tu visión", placeholder="Ej: Rostro de una mujer cyborg bajo la lluvia neón")
            btn = gr.Button("🚀 GENERAR PRODUCCIÓN", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="Prompt Técnico Generado")
            output_video = gr.Video(label="Video Producido")

    btn.click(pipeline_maestro, inputs=[input_text], outputs=[output_text, output_video])

if __name__ == "__main__":
    demo.launch(share=True)
