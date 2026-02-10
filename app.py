import gradio as gr
import os, subprocess, time, torch
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image
from PIL import Image

# 1. Configuración de Modelos (Carga optimizada para GPU T4)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Generador de Imágenes (Crea la base: rostros, laboratorios, paisajes)
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device)

# Generador de Video (Anima la imagen con coherencia temporal)
pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device)

# --- MAGIA PARA EVITAR EL ERROR DE MEMORIA ---
pipe_vid.enable_model_cpu_offload()
pipe_vid.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe_vid.vae.enable_tiling()

def pipeline_autonomo(prompt_usuario):
    base_dir = os.getcwd()
    video_out = os.path.join(base_dir, "VCPI_AI_Production.mp4")
    
    # 1. DIRECTOR IA (Ollama): Expande el prompt para realismo extremo
    try:
        # Forzamos a Llama3 a ser un experto en fotografía cinematográfica
        query = f"Actúa como director de cine. Mejora este prompt para una IA de video: '{prompt_usuario}'. Describe iluminación, textura de piel o materiales y atmósfera en 20 palabras."
        prompt_pro = subprocess.check_output(["ollama", "run", "llama3", query], timeout=30).decode('utf-8').strip()
    except:
        prompt_pro = f"Cinematic shot of {prompt_usuario}, 8k, highly detailed, masterwork."

    # 2. ESCENÓGRAFO (SDXL): Genera la imagen maestra
    print("🎨 Creando visuales de alta fidelidad...")
    image = pipe_img(prompt=prompt_pro, num_inference_steps=2, guidance_scale=0.0).images[0]
    image = image.resize((512, 512)) # Tamaño ideal para estabilidad
    img_path = os.path.join(base_dir, "master_frame.png")
    image.save(img_path)

    # 3. ANIMADOR (SVD): Genera el video (14 cuadros para evitar crash)
    print("🎬 Animando escena (Nivel Sora)...")
    try:
        # Reducimos el decode_chunk_size a 2 para no saturar la VRAM
        frames = pipe_vid(
            image, 
            decode_chunk_size=2, 
            motion_bucket_id=127, 
            num_frames=14, 
            fps=7
        ).frames[0]
        
        # Guardar frames para ensamblaje
        frame_dir = os.path.join(base_dir, "temp_frames")
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(f"{frame_dir}/{i:03d}.png")
            
    except RuntimeError as e:
        return f"Error de memoria: {e}. Intenta con un prompt más simple.", None

    # 4. AUDIO (Voz y Fondo)
    voz_path = os.path.join(base_dir, "voz.mp3")
    subprocess.run(["edge-tts", "--text", prompt_pro[:120], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])

    # 5. ENSAMBLAJE FINAL (FFMPEG)
    # Crea el video uniendo frames y audio
    subprocess.run(
        f"ffmpeg -framerate 7 -i {frame_dir}/%03d.png -i {voz_path} -c:v libx264 -pix_fmt yuv420p -shortest {video_out} -y",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    return prompt_pro, video_out

# --- INTERFAZ GRADIO ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="cyan")) as demo:
    gr.Markdown("# 🌌 VCPI v5.0: Generación Autónoma Pro")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Idea Base (Ej: Rostro de guerrero ciborg, laboratorio de alquimia)")
            btn = gr.Button("GENERAR PRODUCCIÓN COMPLETA", variant="primary")
        with gr.Column():
            out_v = gr.Video(label="Resultado IA (Sora Style)")
            out_t = gr.Textbox(label="Prompt Expandido por el Director")

    btn.click(pipeline_autonomo, inputs=[idea], outputs=[out_t, out_v])

demo.launch(share=True)
