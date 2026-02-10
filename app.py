import gradio as gr
import os, subprocess, time
import soundfile as sf
import numpy as np
from motor_vcpi import MotorVCPI

def pipeline_maestro(prompt, duracion):
    # Definir rutas absolutas para evitar confusiones
    base_path = os.getcwd()
    voz_path = os.path.join(base_path, "voz.mp3")
    musica_path = os.path.join(base_path, "musica.wav")
    video_out = os.path.join(base_path, "VCPI_Final.mp4")
    
    ollama_bin = os.getenv('OLLAMA_BIN', 'ollama')
    
    # 1. Guion
    try:
        guion = subprocess.check_output([ollama_bin, "run", "llama3", f"Short sentence (10 words): {prompt}"], timeout=30).decode('utf-8').strip()
    except:
        guion = "System online. Generating visualization for: " + prompt

    # 2. Generar Voz
    # Usamos comillas dobles para el texto para evitar errores con caracteres especiales
    subprocess.run(["edge-tts", "--text", guion, "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])
    
    # 3. Generar Música (Silencio de seguridad)
    sf.write(musica_path, np.zeros(32000 * int(duracion)), 32000)

    # 4. Render 3D
    try:
        motor = MotorVCPI()
        img_render = motor.crear_escena() # motor_vcpi.py debe retornar el path absoluto
    except Exception as e:
        print(f"Error 3D: {e}")
        img_render = os.path.join(base_path, "fallback.png")

    # 5. Esperar un segundo para asegurar que el disco escribió los archivos
    time.sleep(2)

    # 6. FFMPEG con rutas verificadas
    if os.path.exists(voz_path) and os.path.exists(img_render):
        cmd = (
            f'ffmpeg -loop 1 -i "{img_render}" -i "{musica_path}" -i "{voz_path}" '
            f'-filter_complex "[1:a][2:a]amix=inputs=2:duration=first" '
            f'-c:v libx264 -t {duracion} -pix_fmt yuv420p "{video_out}" -y'
        )
        subprocess.run(cmd, shell=True)
    else:
        return "Error: No se generaron los archivos necesarios.", None

    return guion, video_out

# Interfaz
with gr.Blocks() as demo:
    gr.Markdown("# 🌌 VCPI - Render Directo")
    idea = gr.Textbox(label="Prompt")
    btn = gr.Button("GENERAR VIDEO")
    out_t = gr.Textbox(label="Guion")
    out_v = gr.Video(label="Resultado")
    
    btn.click(pipeline_maestro, [idea, gr.State(10)], [out_t, out_v])

demo.launch(share=True)
