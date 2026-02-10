import gradio as gr
import os, subprocess, time, torch, shutil
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image

# Asignación de dispositivos
gpu_0 = "cuda:0"
gpu_1 = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"

print(f"🚀 Usando {gpu_0} para Imágenes y {gpu_1} para Video")

# 1. Carga de modelos distribuida
pipe_img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to(gpu_0)

pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
).to(gpu_1)

# Optimizaciones críticas para GPU 1 (Video)
pipe_vid.vae.to(torch.float32) # Evita errores de precisión
pipe_vid.enable_sequential_cpu_offload(device=gpu_1) 

def generar_vcpi_dual_gpu(prompt_usuario):
    base_dir = os.getcwd()
    video_out = os.path.join(base_dir, "VCPI_DualGPU_5s.mp4")
    
    # 1. Director IA (Ollama)
    query = f"Cinematic prompt for: {prompt_usuario}. High detail, stable anatomy, full body, 8k."
    prompt_ai = subprocess.check_output(["ollama", "run", "llama3", query]).decode('utf-8').strip()

    # 2. Imagen Maestra (En GPU 0)
    image = pipe_img(prompt=prompt_ai, num_inference_steps=4, guidance_scale=1.2).images[0]
    image = image.resize((512, 512))
    image.save("master.png")

    # 3. Animación (En GPU 1)
    print("🎬 Animando 25 cuadros en GPU secundaria...")
    try:
        # Forzamos la ejecución en la GPU 1 para no saturar la 0
        output = pipe_vid(
            image, 
            num_frames=25, 
            motion_bucket_id=60, 
            noise_aug_strength=0.04,
            decode_chunk_size=1
        )
        frames = output.frames[0]
    except Exception as e:
        return f"Error en GPU: {e}", None

    # 4. Procesamiento de archivos
    frame_dir = "temp_frames"
    if os.path.exists(frame_dir): shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(f"{frame_dir}/{i:03d}.png")

    # 5. Voz y Video Final
    voz_path = "voz.mp3"
    subprocess.run(["edge-tts", "--text", prompt_ai[:150], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])
    subprocess.run(
        f"ffmpeg -framerate 5 -i {frame_dir}/%03d.png -i {voz_path} -c:v libx264 -pix_fmt yuv420p -shortest {video_out} -y",
        shell=True, capture_output=True
    )
    
    return prompt_ai, video_out

# Interfaz
with gr.Blocks(theme=gr.themes.Default(primary_hue="orange")) as demo:
    gr.Markdown("# 🌌 VCPI v8.0 - Dual GPU Edition (5s Coherente)")
    with gr.Row():
        idea = gr.Textbox(label="Instrucción (Ej: Guerrero samurái en el espacio)")
        btn = gr.Button("GENERAR PRODUCCIÓN DUAL-GPU", variant="primary")
    with gr.Row():
        out_p = gr.Textbox(label="Prompt de Llama3")
        out_v = gr.Video(label="Video (GPU 1)")

    btn.click(generar_vcpi_dual_gpu, inputs=[idea], outputs=[out_p, out_v])

demo.launch(share=True)
