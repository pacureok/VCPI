import gradio as gr
import os
import subprocess
import torch
import numpy as np
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")

# --- CARGA DE MOTORES ---
try:
    from audiocraft.models import musicgen
    AUDIO_ENGINE = True
except:
    AUDIO_ENGINE = False

try:
    from motor_vcpi import MotorVCPI
    motor = MotorVCPI()
except:
    motor = None

def pipeline_vcpi(prompt, duracion):
    # 1. GUION (Ollama)
    ollama_bin = os.getenv('OLLAMA_BIN', '/usr/local/bin/ollama')
    try:
        guion = subprocess.check_output([ollama_bin, "run", "llama3", f"Frase corta de 10 palabras sobre: {prompt}"], timeout=30).decode('utf-8')
    except:
        guion = f"Protocolo Pacure Labs activado: {prompt}"

    # 2. VOZ (Edge-TTS)
    voz_path = "narracion.mp3"
    os.system(f'edge-tts --text "{guion}" --write-media {voz_path} --voice es-MX-DaliaNeural')

    # 3. MÚSICA (MusicGen Small)
    musica_path = "musica.wav"
    if AUDIO_ENGINE:
        try:
            model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small')
            model.set_generation_params(duration=int(duracion))
            res = model.generate([f"Dark ambient, cinematic, {prompt}"])
            sf.write(musica_path, res[0, 0].cpu().numpy(), 32000)
        except:
            sf.write(musica_path, np.zeros(32000 * int(duracion)), 32000)
    else:
        sf.write(musica_path, np.zeros(32000 * int(duracion)), 32000)

    # 4. RENDER 3D (Panda3D)
    img_render = motor.crear_escena() if motor else "fallback.png"

    # 5. ENSAMBLAJE (FFMPEG)
    video_out = "VCPI_Final.mp4"
    # Mezcla: imagen + música de fondo + voz del narrador
    ffmpeg_cmd = (
        f'ffmpeg -loop 1 -i {img_render} -i {musica_path} -i {voz_path} '
        f'-filter_complex "[1:a][2:a]amix=inputs=2:duration=first" '
        f'-c:v libx264 -t {duracion} -pix_fmt yuv420p {video_out} -y'
    )
    os.system(ffmpeg_cmd)

    return guion, video_out, musica_path

# --- INTERFAZ GRADIO ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌌 VCPI - Generador Multimodal")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Prompt / Idea")
            tiempo = gr.Slider(5, 60, value=15, label="Segundos")
            btn = gr.Button("GENERAR UNIVERSO", variant="primary")
        with gr.Column():
            res_txt = gr.Textbox(label="Guion Generado")
            res_vid = gr.Video(label="Película Final")
    
    btn.click(pipeline_vcpi, [idea, tiempo], [res_txt, res_vid])

demo.launch(share=True)
