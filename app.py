import gradio as gr
import os, subprocess, time, torch, shutil
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image

# Forzar detección de ambas GPUs
device_0 = "cuda:0"
device_1 = "cuda:1"

# 1. Carga de modelos balanceada
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to(device_0)

pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device_1)

# BALANCEO: La GPU 0 ayuda decodificando los cuadros
pipe_vid.vae.to(device_0)
pipe_vid.vae.to(torch.float32)
pipe_vid.enable_sequential_cpu_offload(device=device_1)

def produccion_final(prompt_usuario):
    # RUTAS ABSOLUTAS (Evita el error de Gradio)
    base_path = "/kaggle/working"
    video_out = os.path.join(base_path, "resultado_final.mp4")
    frame_dir = os.path.join(base_path, "frames_temp")
    
    if os.path.exists(frame_dir): shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    # 1. Llama 3
    query = f"Cinematic director: Create a highly detailed 5s scene prompt for {prompt_usuario}. Stable anatomy, 8k."
    prompt_ai = subprocess.check_output(["ollama", "run", "llama3", query]).decode('utf-8').strip()

    # 2. Generar Imagen (GPU 0)
    img_path = os.path.join(base_path, "master.png")
    image = pipe_img(prompt=prompt_ai, num_inference_steps=4, guidance_scale=1.2).images[0]
    image = image.resize((512, 512))
    image.save(img_path)

    # 3. Generar Video (GPU 1 Unet + GPU 0 VAE)
    print("🎬 Generando frames con balanceo 90/50...")
    output = pipe_vid(
        image, 
        num_frames=25, 
        motion_bucket_id=70, 
        noise_aug_strength=0.04,
        decode_chunk_size=1
    )
    
    # 4. Guardar Frames
    for i, frame in enumerate(output.frames[0]):
        frame.save(os.path.join(frame_dir, f"{i:03d}.png"))

    # 5. Audio y FFMPEG
    voz_path = os.path.join(base_path, "voz.mp3")
    subprocess.run(["edge-tts", "--text", prompt_ai[:100], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])
    
    # Comando FFMPEG con perfil de compatibilidad web
    ff_cmd = [
        "ffmpeg", "-framerate", "5", "-i", f"{frame_dir}/%03d.png",
        "-i", voz_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", video_out, "-y"
    ]
    subprocess.run(ff_cmd, capture_output=True)
    
    time.sleep(1) # Espera de seguridad para el sistema de archivos
    return prompt_ai, video_out

# Interfaz
with gr.Blocks() as demo:
    gr.Markdown("# 🌌 VCPI v9.5 - Dual GPU Stable")
    with gr.Row():
        in_idea = gr.Textbox(label="Instrucción")
        btn = gr.Button("GENERAR 5S", variant="primary")
    with gr.Row():
        out_txt = gr.Textbox(label="Prompt")
        out_vid = gr.Video(label="Video")

    btn.click(produccion_final, inputs=[in_idea], outputs=[out_txt, out_vid])

demo.launch(share=True)
