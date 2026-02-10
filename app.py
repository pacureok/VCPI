import gradio as gr
import os, subprocess, torch, site
import soundfile as sf
import numpy as np
from motor_vcpi import MotorVCPI

# Configuración de entorno
os.environ['PANDA_PRC_SET_window_type'] = 'offscreen'

def pipeline_maestro(prompt, duracion):
    ollama_bin = os.getenv('OLLAMA_BIN', 'ollama')
    
    # 1. TEXTO
    try:
        guion = subprocess.check_output([ollama_bin, "run", "llama3", f"Frase de 10 palabras: {prompt}"], timeout=30).decode('utf-8')
    except:
        guion = "Error en Ollama. Iniciando renderizado de emergencia."

    # 2. AUDIO (Música + Voz)
    voz_path = "voz.mp3"
    os.system(f'edge-tts --text "{guion}" --write-media {voz_path} --voice es-MX-DaliaNeural')
    
    # Música (Silencio si falla MusicGen para no detener el proceso)
    musica_path = "musica.wav"
    sf.write(musica_path, np.zeros(32000 * int(duracion)), 32000)

    # 3. VIDEO (Motor 3D)
    try:
        motor = MotorVCPI()
        img_path = motor.crear_escena()
    except Exception as e:
        print(f"Error Motor: {e}")
        img_path = "render_3d.png" # Path por defecto

    # 4. ENSAMBLAJE FINAL
    video_out = "VCPI_Final.mp4"
    cmd = f'ffmpeg -loop 1 -i {img_path} -i {musica_path} -i {voz_path} -filter_complex "[1:a][2:a]amix=inputs=2:duration=first" -c:v libx264 -t {duracion} -pix_fmt yuv420p {video_out} -y'
    os.system(cmd)

    return guion, video_out

# Interfaz
with gr.Blocks() as demo:
    idea = gr.Textbox(label="Instrucción")
    dur = gr.Slider(5, 30, value=10)
    btn = gr.Button("GENERAR")
    out_v = gr.Video()
    out_t = gr.Textbox()
    btn.click(pipeline_maestro, [idea, dur], [out_t, out_v])

demo.launch(share=True)
