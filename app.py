import gradio as gr
import os, subprocess, time, torch, shutil
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image

# Configuración Dual-GPU balanceada
device_img = "cuda:0"
device_vid = "cuda:1"

# Carga de modelos
pipe_img = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to(device_img)
pipe_vid = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16").to(device_vid)

# Balanceo: GPU 0 ayuda con el VAE
pipe_vid.vae.to(device_img)
pipe_vid.enable_sequential_cpu_offload(device=device_vid)

def generar_solo_mp4(prompt_usuario):
    # Rutas fijas y absolutas
    video_path = "/kaggle/working/produccion_final.mp4"
    frame_dir = "/kaggle/working/frames_temp"
    
    if os.path.exists(frame_dir): shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    # 1. Imagen (GPU 0)
    image = pipe_img(prompt=prompt_usuario, num_inference_steps=4).images[0].resize((512, 512))
    
    # 2. Video (GPU 1 + GPU 0)
    # decode_chunk_size=1 es vital para que no explote la memoria
    output = pipe_vid(image, num_frames=25, motion_bucket_id=70, noise_aug_strength=0.04, decode_chunk_size=1)
    
    # 3. Guardar frames rápido
    for i, frame in enumerate(output.frames[0]):
        frame.save(f"{frame_dir}/{i:03d}.png")

    # 4. FFMPEG Directo (Sin audio para velocidad máxima y evitar errores de codec)
    # 5fps = 5 segundos exactos
    subprocess.run(
        f"ffmpeg -framerate 5 -i {frame_dir}/%03d.png -c:v libx264 -pix_fmt yuv420p -movflags +faststart {video_path} -y",
        shell=True, capture_output=True
    )
    
    return video_path

# Interfaz minimalista
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 VCPI - Generador MP4 Directo")
    with gr.Row():
        input_txt = gr.Textbox(label="Instrucción")
        btn = gr.Button("GENERAR MP4")
    
    # El componente Video de Gradio recibirá la ruta y la cargará
    output_video = gr.Video(label="Resultado MP4")

    btn.click(generar_solo_mp4, inputs=[input_txt], outputs=[output_video])

demo.launch(share=True, show_error=True)
