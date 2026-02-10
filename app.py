import gradio as gr
import os, subprocess, torch, shutil
# Importación directa de la clase para evitar escaneos de módulos fallidos
from diffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipeline
from diffusers import AutoPipelineForText2Image

# Configuración Dual-GPU
gpu_0 = "cuda:0"
gpu_1 = "cuda:1"

# Carga de modelos
print("🎨 Cargando Imagen (GPU 0)...")
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(gpu_0)

print("🎬 Cargando Video (GPU 1)...")
pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(gpu_1)

# Balanceo dinámico: Movemos el VAE a la GPU 0
pipe_vid.vae.to(gpu_0)
pipe_vid.vae.to(torch.float32)
pipe_vid.enable_sequential_cpu_offload(device=gpu_1)

def generar_video_directo(prompt):
    out_mp4 = "/kaggle/working/produccion_final.mp4"
    tmp_frames = "/kaggle/working/frames_temp"
    
    if os.path.exists(tmp_frames): shutil.rmtree(tmp_frames)
    os.makedirs(tmp_frames, exist_ok=True)

    # 1. Imagen (GPU 0)
    image = pipe_img(prompt=prompt, num_inference_steps=2).images[0].resize((512, 512))
    
    # 2. Video 5s (GPU 1 + Ayuda GPU 0)
    print("🎬 Generando 25 frames...")
    output = pipe_vid(
        image, 
        num_frames=25, 
        motion_bucket_id=70, 
        noise_aug_strength=0.04, 
        decode_chunk_size=1
    )
    
    # 3. Guardar frames y ensamblar
    for i, frame in enumerate(output.frames[0]):
        frame.save(f"{tmp_frames}/{i:03d}.png")

    subprocess.run(
        f"ffmpeg -framerate 5 -i {tmp_frames}/%03d.png -c:v libx264 -pix_fmt yuv420p -movflags +faststart {out_mp4} -y",
        shell=True, capture_output=True
    )
    
    return out_mp4

# Interfaz
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 VCPI v10 - Zero Import Errors")
    with gr.Row():
        txt = gr.Textbox(label="Instrucción Visual")
        btn = gr.Button("GENERAR MP4")
    
    vid = gr.Video(label="Resultado Final")
    btn.click(generar_video_directo, inputs=[txt], outputs=[vid])

demo.launch(share=True)
