import gradio as gr
import os
import subprocess
import torch
import gc
import numpy as np
import soundfile as sf
from audiocraft.models import musicgen
from motor_vcpi import MotorVCPI 

# --- LIMPIEZA ---
def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()

# --- CARGA DE MODELOS ---
print("🚀 VCPI: Cargando motores...")
clear_mem()
try:
    # Usamos la versión 'small' si la memoria falla, pero 'melody' es mejor
    music_model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small', device='cuda')
    motor = MotorVCPI()
except Exception as e:
    print(f"Error: {e}")
    motor = None

def pipeline_vcpi(prompt, duracion):
    clear_mem()
    
    # 1. Guion (Ollama)
    guion = subprocess.getoutput(f"ollama run llama3 'Frase corta de terror liminal: {prompt}'")

    # 2. Música (MusicGen)
    music_model.set_generation_params(duration=int(duracion))
    res = music_model.generate([f"Dark ambient, backrooms, {prompt}"], progress=True)
    audio_data = res.cpu().numpy()[0, 0]
    sf.write("audio.wav", audio_data, 32000)

    # 3. Voz (TTS)
    os.system(f'edge-tts --text "{guion}" --write-media voz.mp3 --voice es-MX-DaliaNeural')

    # 4. 3D Render
    img_render = motor.crear_escena() if motor else "fallback.png"

    # 5. Video Final (FFMPEG)
    video_out = "VCPI_Final.mp4"
    os.system(f'ffmpeg -loop 1 -i {img_render} -i audio.wav -i voz.mp3 -filter_complex "[1:a][2:a]amix=inputs=2:duration=first" -c:v libx264 -t {duracion} -pix_fmt yuv420p {video_out} -y')

    return guion, video_out, "audio.wav"

# --- INTERFAZ ---
with gr.Blocks() as demo:
    gr.Markdown("# 🌌 VCPI Multimedia Hub")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Idea")
            tiempo = gr.Slider(10, 180, value=30, label="Segundos")
            btn = gr.Button("GENERAR")
        with gr.Column():
            txt = gr.Textbox(label="Guion")
            vid = gr.Video(label="Video MP4")
            aud = gr.Audio(label="Banda Sonora")

    btn.click(pipeline_vcpi, [idea, tiempo], [txt, vid, aud])

demo.launch(share=True)
