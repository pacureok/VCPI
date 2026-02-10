import gradio as gr
import os
import subprocess
import torch
import numpy as np
import soundfile as sf
from audiocraft.models import musicgen
from motor_vcpi import MotorVCPI 

# Limpieza de memoria
torch.cuda.empty_cache()

# Carga de motores
print("🚀 VCPI: Iniciando motores...")
try:
    # Usamos 'small' para evitar que Kaggle se quede sin memoria (VRAM)
    music_model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small', device='cuda')
    motor = MotorVCPI()
except Exception as e:
    print(f"Error cargando modelos: {e}")
    motor = None

def pipeline_vcpi(prompt, duracion):
    # 1. Guion (Ollama)
    # Forzamos la ruta completa por seguridad
    guion = subprocess.getoutput(f"/usr/local/bin/ollama run llama3 'Frase corta de exploracion: {prompt}'")

    # 2. Música (MusicGen)
    music_model.set_generation_params(duration=int(duracion))
    res = music_model.generate([f"Dark ambient, cinematic, {prompt}"], progress=True)
    audio_data = res.cpu().numpy()[0, 0]
    sf.write("audio_bg.wav", audio_data, 32000)

    # 3. Voz (TTS)
    os.system(f'edge-tts --text "{guion}" --write-media voz_narrador.mp3 --voice es-MX-DaliaNeural')

    # 4. Render 3D
    img_render = motor.crear_escena() if motor else "fallback.png"

    # 5. Ensamblaje Final (Video MP4)
    video_out = "VCPI_Pelicula.mp4"
    # Mezclamos la imagen fija con el audio de fondo y la voz narrada
    ffmpeg_cmd = (
        f'ffmpeg -loop 1 -i {img_render} -i audio_bg.wav -i voz_narrador.mp3 '
        f'-filter_complex "[1:a][2:a]amix=inputs=2:duration=first" '
        f'-c:v libx264 -t {duracion} -pix_fmt yuv420p {video_out} -y'
    )
    os.system(ffmpeg_cmd)

    return guion, video_out, "audio_bg.wav"

# Interfaz Gradio
with gr.Blocks(title="VCPI Hub") as demo:
    gr.Markdown("# 🌌 VCPI - Sistema de Cine IA")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Tu idea (Prompt)")
            tiempo = gr.Slider(10, 120, value=30, label="Segundos de video")
            btn = gr.Button("GENERAR MUNDO")
        with gr.Column():
            txt = gr.Textbox(label="Guion Generado")
            vid = gr.Video(label="Pelicula Final")
            aud = gr.Audio(label="Banda Sonora")

    btn.click(pipeline_vcpi, [idea, tiempo], [txt, vid, aud])

demo.launch(share=True)
