import gradio as gr
import os
import subprocess
import torch
import warnings

# Silenciamos las alertas de versiones de Audiocraft
warnings.filterwarnings("ignore")

# Intentamos cargar Audiocraft de forma segura
try:
    from audiocraft.models import musicgen
    import numpy as np
    import soundfile as sf
except Exception as e:
    print(f"⚠️ Error al importar audio: {e}")

from motor_vcpi import MotorVCPI 

def pipeline_vcpi(prompt, duracion):
    # 1. GUION (Ollama)
    # Si Ollama tarda, le ponemos un timeout
    print(f"🎬 Generando guion para: {prompt}")
    try:
        guion = subprocess.check_output(["ollama", "run", "llama3", f"Escribe una frase de 10 palabras sobre: {prompt}"], timeout=30).decode('utf-8')
    except:
        guion = "En el vacío del tiempo, el monolito espera."

    # 2. MÚSICA (MusicGen Small para evitar crashes)
    print("🎵 Generando banda sonora...")
    model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small')
    model.set_generation_params(duration=int(duracion))
    res = model.generate([f"Dark ambient, cinematic, {prompt}"])
    audio_path = "output_bg.wav"
    sf.write(audio_path, res[0, 0].cpu().numpy(), 32000)

    # 3. VOZ (Edge-TTS)
    print("🎙️ Generando voz...")
    voz_path = "output_voz.mp3"
    os.system(f'edge-tts --text "{guion}" --write-media {voz_path} --voice es-MX-DaliaNeural')

    # 4. RENDER 3D (Panda3D)
    print("🎥 Renderizando 3D...")
    motor = MotorVCPI()
    img_render = motor.crear_escena()

    # 5. ENSAMBLAJE (FFMPEG)
    video_out = "VCPI_Pelicula_Final.mp4"
    os.system(f'ffmpeg -loop 1 -i {img_render} -i {audio_path} -i {voz_path} -filter_complex "[1:a][2:a]amix=inputs=2:duration=first" -c:v libx264 -t {duracion} -pix_fmt yuv420p {video_out} -y')

    return guion, video_out, audio_path

# Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🌌 VCPI Multimedia Engine")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Instrucción")
            sec = gr.Slider(10, 60, value=20, label="Segundos")
            btn = gr.Button("GENERAR")
        with gr.Column():
            res_v = gr.Video()
            res_t = gr.Textbox()
    
    btn.click(pipeline_vcpi, [idea, sec], [res_t, res_v])

demo.launch(share=True)
