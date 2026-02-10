import gradio as gr
import os, subprocess, time, torch, shutil
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image

# Configuración Dual-GPU
# GPU 0: Imágenes (SDXL) y Decodificación VAE
# GPU 1: Animación Temporal (SVD)
device_img = "cuda:0"
device_vid = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"

print(f"💎 Primaria (0): {torch.cuda.get_device_name(0)}")
if torch.cuda.device_count() > 1:
    print(f"💎 Secundaria (1): {torch.cuda.get_device_name(1)}")

# 1. Carga Distribuida
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to(device_img)

pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
).to(device_vid)

# Optimizaciones de memoria
pipe_vid.enable_sequential_cpu_offload(device=device_vid)
pipe_vid.vae.to(torch.float32) # Estabilidad anatómica

def produccion_dual(prompt_usuario):
    # Rutas absolutas para evitar que Gradio se pierda
    base_dir = "/kaggle/working/VCPI"
    video_out = os.path.join(base_dir, "output_produccion.mp4")
    frame_dir = os.path.join(base_dir, "temp_frames")
    
    if os.path.exists(frame_dir): shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    # 1. Llama3: Director de Arte
    query = f"Cinematic art director: Expand this into a technical prompt for video: {prompt_usuario}. Detailed anatomy, full limbs, 8k, photorealistic."
    prompt_ai = subprocess.check_output(["ollama", "run", "llama3", query]).decode('utf-8').strip()

    # 2. GPU 0: Generar Imagen (Rápido)
    print("🎨 [GPU 0] Generando imagen base...")
    image = pipe_img(prompt=prompt_ai, num_inference_steps=4, guidance_scale=1.2).images[0]
    image = image.resize((512, 512))
    image.save("master.png")

    # 3. GPU 1: Generar Video (Pesado)
    print("🎬 [GPU 1] Animando 25 cuadros (5 segundos)...")
    output = pipe_vid(
        image, 
        num_frames=25, 
        motion_bucket_id=70, 
        noise_aug_strength=0.04,
        decode_chunk_size=1 # Estabilidad total
    )
    frames = output.frames[0]

    # 4. Guardar Frames
    for i, frame in enumerate(frames):
        frame.save(f"{frame_dir}/{i:03d}.png")

    # 5. Audio y Ensamblaje Final
    print("🎥 Ensamblando con FFMPEG...")
    voz_path = os.path.join(base_dir, "voz.mp3")
    subprocess.run(["edge-tts", "--text", prompt_ai[:150], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])
    
    # Comando FFMPEG optimizado para compatibilidad con navegadores
    subprocess.run(
        f"ffmpeg -framerate 5 -i {frame_dir}/%03d.png -i {voz_path} -c:v libx264 -pix_fmt yuv420p -profile:v baseline -level 3.0 -movflags +faststart -shortest {video_out} -y",
        shell=True, capture_output=True
    )
    
    return prompt_ai, video_out

# Interfaz
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌌 VCPI v8.5 - Dual-GPU 5s Edition")
    with gr.Row():
        input_t = gr.Textbox(label="Instrucción Visual")
        btn = gr.Button("GENERAR PRODUCCIÓN DUAL", variant="primary")
    with gr.Row():
        output_t = gr.Textbox(label="Prompt Técnico (Llama3)")
        output_v = gr.Video(label="Video Final (5 seg @ 5fps)")

    btn.click(produccion_dual, inputs=[input_t], outputs=[output_t, output_v])

demo.launch(share=True)
