import gradio as gr
import os
import subprocess
import torch
import soundfile as sf
import numpy as np

# --- CARGA SEGURA DE MUSICGEN ---
try:
    from audiocraft.models import musicgen
    print("✅ MusicGen cargado.")
except Exception as e:
    print(f"⚠️ Error cargando MusicGen: {e}")
    musicgen = None

from motor_vcpi import MotorVCPI 

def pipeline_maestro(prompt, duracion):
    # 1. GUION (Ollama con Failsafe)
    ollama_bin = os.getenv('OLLAMA_BIN', '/usr/local/bin/ollama')
    print("📝 Generando texto...")
    try:
        guion = subprocess.check_output([ollama_bin, "run", "llama3", f"Crea una frase de 10 palabras sobre: {prompt}"], timeout=25).decode('utf-8')
    except:
        guion = "Iniciando protocolo Pacure Labs. El sistema evoluciona."

    # 2. VOZ (Edge-TTS)
    print("🎙️ Generando voces...")
    voz_path = "voz.mp3"
    os.system(f'edge-tts --text "{guion}" --write-media {voz_path} --voice es-MX-DaliaNeural')

    # 3. MÚSICA (MusicGen con manejo de errores)
    print("🎵 Generando banda sonora...")
    musica_path = "musica.wav"
    try:
        if musicgen:
            model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small')
            model.set_generation_params(duration=int(duracion))
            res = model.generate([f"Futuristic cyberpunk, {prompt}"])
            sf.write(musica_path, res[0, 0].cpu().numpy(), 32000)
        else:
            raise Exception("No engine")
    except:
        sf.write(musica_path, np.zeros(32000 * int(duracion)), 32000)

    # 4. RENDER 3D
    print("🎥 Capturando render 3D...")
    try:
        motor = MotorVCPI()
        img_path = motor.crear_escena()
    except:
        img_path = "fallback.png"

    # 5. ENSAMBLAJE (FFMPEG)
    video_out = "VCPI_Final.mp4"
    ffmpeg_cmd = (
        f'ffmpeg -loop 1 -i {img_path} -i {musica_path} -i {voz_path} '
        f'-filter_complex "[1:a][2:a]amix=inputs=2:duration=first" '
        f'-c:v libx264 -t {duracion} -pix_fmt yuv420p {video_out} -y'
    )
    subprocess.run(ffmpeg_cmd, shell=True)

    return guion, video_out, musica_path

# --- UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🌌 VCPI Pacure Labs v3.0")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Instrucción creativa")
            tiempo = gr.Slider(5, 60, value=15, label="Duración del clip")
            btn = gr.Button("GENERAR TODO", variant="primary")
        with gr.Column():
            res_v = gr.Video(label="Video Multimodal")
            res_t = gr.Textbox(label="Guion IA")

    btn.click(pipeline_maestro, [idea, tiempo], [res_t, res_v])

demo.launch(share=True)
