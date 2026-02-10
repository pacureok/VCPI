import gradio as gr
import os, subprocess, time, torch, shutil
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Modelos con carga optimizada
pipe_img = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to(device)
pipe_vid = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16").to(device)

# Optimizaciones de memoria para 25 frames
pipe_vid.enable_model_cpu_offload()
pipe_vid.vae.enable_slicing()

def generar_produccion_pro(prompt_usuario):
    base_dir = os.getcwd()
    video_out = os.path.join(base_dir, "VCPI_Final_5s.mp4")
    
    # 1. Director IA (Llama3): Enfocado en anatomía
    query = f"Act as a film director. Create a high-detail prompt for: {prompt_usuario}. Mention full body, clear anatomical details, stable limbs, and cinematic lighting. 25 words max."
    prompt_ai = subprocess.check_output(["ollama", "run", "llama3", query]).decode('utf-8').strip()

    # 2. Imagen Base: Más pasos = mejores manos y pies
    image = pipe_img(prompt=prompt_ai, num_inference_steps=4, guidance_scale=1.2).images[0]
    image = image.resize((512, 512))
    image.save("master.png")

    # 3. Video: 25 frames con baja fuerza de ruido para evitar deformaciones
    print("🎬 Generando video de 5 segundos con coherencia...")
    frames = pipe_vid(
        image, 
        decode_chunk_size=2, 
        num_frames=25, 
        motion_bucket_id=80,      # Movimiento controlado para no romper extremidades
        noise_aug_strength=0.08   # Mantiene la fidelidad a la imagen original
    ).frames[0]
    
    frame_dir = "temp_frames"
    if os.path.exists(frame_dir): shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(f"{frame_dir}/{i:03d}.png")

    # 4. Voz
    voz_path = "voz.mp3"
    subprocess.run(["edge-tts", "--text", prompt_ai[:150], "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])

    # 5. FFMPEG: 5 fps = 5 segundos exactos (25 frames / 5 fps)
    subprocess.run(
        f"ffmpeg -framerate 5 -i {frame_dir}/%03d.png -i {voz_path} -c:v libx264 -pix_fmt yuv420p -shortest {video_out} -y",
        shell=True, capture_output=True
    )
    
    return prompt_ai, video_out

# Interfaz Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌌 VCPI v7.0: Coherencia y Producción 5s")
    with gr.Row():
        idea = gr.Textbox(label="Instrucción (Ej: Astronauta caminando en Marte de cuerpo completo)")
        btn = gr.Button("GENERAR VIDEO 5S", variant="primary")
    with gr.Row():
        out_p = gr.Textbox(label="Prompt Técnico")
        out_v = gr.Video(label="Producción Final")
    
    btn.click(generar_produccion_pro, inputs=[idea], outputs=[out_p, out_v])

demo.launch(share=True)
