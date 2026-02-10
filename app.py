import gradio as gr
import os, subprocess, torch
import soundfile as sf
import numpy as np
from motor_vcpi import MotorVCPI

def pipeline_maestro(prompt, duracion):
    ollama_bin = os.getenv('OLLAMA_BIN', 'ollama')
    
    # 1. Guion
    try:
        guion = subprocess.check_output([ollama_bin, "run", "llama3", f"Frase corta de 10 palabras: {prompt}"], timeout=30).decode('utf-8')
    except:
        guion = "Conexión establecida. Iniciando visualización."

    # 2. Voz y Música (Failsafe)
    os.system(f'edge-tts --text "{guion}" --write-media voz.mp3 --voice es-MX-DaliaNeural')
    sf.write("musica.wav", np.zeros(32000 * int(duracion)), 32000)

    # 3. Render 3D
    try:
        motor = MotorVCPI()
        img_path = motor.crear_escena()
    except Exception as e:
        print(f"Error 3D: {e}")
        img_path = "fallback.png"

    # 4. FFMPEG (Ensamblaje Robusto)
    video_out = "VCPI_Final.mp4"
    os.system(f'ffmpeg -loop 1 -i {img_path} -i musica.wav -i voz.mp3 -filter_complex "[1:a][2:a]amix=inputs=2:duration=first" -c:v libx264 -t {duracion} -pix_fmt yuv420p {video_out} -y')

    return guion, video_out

with gr.Blocks() as demo:
    gr.Markdown("# 🌌 VCPI v4.0 (Fixed Syntax)")
    idea = gr.Textbox(label="Prompt")
    btn = gr.Button("GENERAR")
    out_v = gr.Video()
    btn.click(pipeline_maestro, [idea, gr.State(10)], [gr.State(), out_v])

demo.launch(share=True)
