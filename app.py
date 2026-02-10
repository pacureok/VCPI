import gradio as gr
import os, subprocess, torch, shutil
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image

# Configuración Dual-GPU
device_0 = "cuda:0"
device_1 = "cuda:1"

# Carga simplificada para evitar errores de módulos secundarios
print("🎨 Cargando Generador de Imagen (GPU 0)...")
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device_0)

print("🎬 Cargando Generador de Video (GPU 1)...")
pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device_1)

# Balanceo: El VAE vive en la GPU 0 para apoyar a la 1
pipe_vid.vae.to(device_0)
pipe_vid.enable_sequential_cpu_offload(device=device_1)

def generar_mp4_seguro(prompt_usuario):
    # Rutas absolutas
    out_mp4 = "/kaggle/working/produccion_final.mp4"
    tmp_frames = "/kaggle/working/frames_temp"
    
    if os.path.exists(tmp_frames): shutil.rmtree(tmp_frames)
    os.makedirs(tmp_frames, exist_ok=True)

    # 1. Imagen base
    image = pipe_img(prompt=prompt_usuario, num_inference_steps=2).images[0].resize((512, 512))
    
    # 2. Video de 5 segundos (25 frames)
    # decode_chunk_size=1 evita que la memoria colapse al final
    print("🎬 Generando frames...")
    output = pipe_vid(
        image, 
        num_frames=25, 
        motion_bucket_id=70, 
        noise_aug_strength=0.04, 
        decode_chunk_size=1
    )
    
    # 3. Guardar frames
    for i, frame in enumerate(output.frames[0]):
        frame.save(f"{tmp_frames}/{i:03d}.png")

    # 4. FFMPEG Ultra-rápido (Solo video)
    # Usamos -y para sobrescribir y faststart para que Gradio lo lea rápido
    subprocess.run(
        f"ffmpeg -framerate 5 -i {tmp_frames}/%03d.png -c:v libx264 -pix_fmt yuv420p -movflags +faststart {out_mp4} -y",
        shell=True, capture_output=True
    )
    
    return out_mp4

# Interfaz
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 VCPI MP4 - Dual GPU Balance")
    with gr.Row():
        txt = gr.Textbox(label="Instrucción (Ej: Astronauta en Marte)")
        btn = gr.Button("GENERAR VIDEO")
    
    out_video = gr.Video(label="Resultado Final")
    btn.click(generar_mp4_seguro, inputs=[txt], outputs=[out_video])

demo.launch(share=True)
