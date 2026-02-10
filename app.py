import gradio as gr
import os
import subprocess
import torch
import numpy as np
import soundfile as sf
# Importamos con manejo de error por si la instalación falló
try:
    from audiocraft.models import musicgen
except ImportError:
    print("❌ Error: Audiocraft no detectado.")

from motor_vcpi import MotorVCPI 

# Limpiar VRAM
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("🚀 VCPI: Iniciando Motores Pro...")
try:
    # 'small' es mucho más rápido y estable en entornos cloud
    music_model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small', device='cuda')
    motor = MotorVCPI()
except Exception as e:
    print(f"⚠️ Aviso: {e}")
    motor = None

def pipeline_vcpi(prompt, duracion):
    # 1. Guion (Ollama ruta absoluta)
    guion = subprocess.getoutput(f"/usr/local/bin/ollama run llama3 'Frase corta de explorador: {prompt}'")

    # 2. Música
    music_model.set_generation_params(duration=int(duracion))
    res = music_model.generate([f"Dark ambient, backrooms style, {prompt}"], progress=True)
    audio_data = res.cpu().numpy()[0, 0]
    sf.write("bg_music.wav", audio_data, 32000)

    # 3. Voz
    os.system(f'edge-tts --text "{guion}" --write-media narracion.mp3 --voice es-MX-DaliaNeural')

    # 4. Render 3D
    img_render = motor.crear_escena() if motor else "fallback.png"

    # 5. FFMPEG Ensamblaje
    video_out = "VCPI_Final.mp4"
    ffmpeg_cmd = (
        f'ffmpeg -loop 1 -i {img_render} -i bg_music.wav -i narracion.mp3 '
        f'-filter_complex "[1:a][2:a]amix=inputs=2:duration=first" '
        f'-c:v libx264 -t {duracion} -pix_fmt yuv420p {video_out} -y'
    )
    os.system(ffmpeg_cmd)

    return guion, video_out, "bg_music.wav"

# Interfaz
with gr.Blocks(title="VCPI Hub") as demo:
    gr.Markdown("# 🌌 VCPI - Sistema Multimedia")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Instrucción")
            tiempo = gr.Slider(10, 60, value=30, label="Segundos")
            btn = gr.Button("GENERAR")
        with gr.Column():
            txt = gr.Textbox(label="Guion")
            vid = gr.Video(label="Video")
            aud = gr.Audio(label="Banda Sonora")

    btn.click(pipeline_vcpi, [idea, tiempo], [txt, vid, aud])

demo.launch(share=True)
