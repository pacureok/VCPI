import gradio as gr
import os
import subprocess
import torch
import numpy as np
import soundfile as sf

# Forzamos compatibilidad de numpy para Audiocraft
try:
    import numpy
    if version.parse(numpy.__version__) >= version.parse("2.0.0"):
        print("⚠️ Warning: Numpy 2.0 detectado, Audiocraft podría fallar.")
except: pass

from motor_vcpi import MotorVCPI 

# Carga de modelos con "Lazy Loading"
print("🚀 VCPI: Despertando sistema...")
music_model = None
motor = None

def cargar_modelos():
    global music_model, motor
    try:
        from audiocraft.models import musicgen
        music_model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small', device='cuda')
        motor = MotorVCPI()
    except Exception as e:
        print(f"Error cargando motores: {e}")

def pipeline_vcpi(prompt, duracion):
    if music_model is None: cargar_modelos()
    
    # 1. Guion (Ollama)
    guion = subprocess.getoutput(f"ollama run llama3 'Escribe una frase corta de explorador sobre: {prompt}'")

    # 2. Música
    music_model.set_generation_params(duration=int(duracion))
    res = music_model.generate([f"Dark ambient, cinematic, {prompt}"], progress=True)
    audio_path = "bg_music.wav"
    sf.write(audio_path, res.cpu().numpy()[0, 0], 32000)

    # 3. Voz (Narrador)
    voz_path = "narracion.mp3"
    os.system(f'edge-tts --text "{guion}" --write-media {voz_path} --voice es-MX-DaliaNeural')

    # 4. Render 3D (Panda3D)
    img_render = motor.crear_escena() if motor else "fallback.png"

    # 5. FFMPEG Ensamblaje Final
    video_out = "VCPI_Final.mp4"
    ffmpeg_cmd = (
        f'ffmpeg -loop 1 -i {img_render} -i {audio_path} -i {voz_path} '
        f'-filter_complex "[1:a][2:a]amix=inputs=2:duration=first" '
        f'-c:v libx264 -t {duracion} -pix_fmt yuv420p {video_out} -y'
    )
    os.system(ffmpeg_cmd)

    return guion, video_out, audio_path

# Interfaz de Usuario
with gr.Blocks(title="VCPI Pacure Labs") as demo:
    gr.Markdown("# 🌌 VCPI - Hub de Generación Multimedia")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Prompt / Idea")
            tiempo = gr.Slider(10, 60, value=30, label="Duración (seg)")
            btn = gr.Button("GENERAR MUNDO", variant="primary")
        with gr.Column():
            txt = gr.Textbox(label="Guion")
            vid = gr.Video(label="Película")
            aud = gr.Audio(label="Música")

    btn.click(pipeline_vcpi, [idea, tiempo], [txt, vid, aud])

demo.launch(share=True)
