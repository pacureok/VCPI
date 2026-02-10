import gradio as gr
import os, subprocess, time
from motor_vcpi import MotorVCPI

def pipeline_maestro(prompt, duracion):
    base_dir = os.getcwd()
    video_out = os.path.join(base_dir, "VCPI_Final.mp4")
    
    # 1. Ollama genera el guion
    ollama_bin = os.getenv('OLLAMA_BIN', '/usr/local/bin/ollama')
    try:
        guion = subprocess.check_output([ollama_bin, "run", "llama3", f"Escribe una frase de 10 palabras sobre: {prompt}"], timeout=30).decode('utf-8').strip()
    except:
        guion = f"Visualizando entorno: {prompt}"

    # 2. Voz (TTS)
    voz_path = os.path.join(base_dir, "voz.mp3")
    subprocess.run(["edge-tts", "--text", guion, "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])

    # 3. Renderizado 3D con ESTILO
    try:
        motor = MotorVCPI()
        # Le pasamos el prompt al motor para que decida colores
        img_render = motor.crear_escena(estilo=prompt.lower())
    except Exception as e:
        print(f"Error 3D: {e}")
        img_render = "render_final.png"

    # 4. Música de fondo (silencio rápido por ahora)
    musica_path = os.path.join(base_dir, "musica.wav")
    os.system(f'ffmpeg -f lavfi -i anullsrc=r=32000:cl=mono -t {duracion} {musica_path} -y')

    # 5. Ensamblaje FFMPEG
    time.sleep(1)
    cmd = (
        f'ffmpeg -loop 1 -i "{img_render}" -i "{musica_path}" -i "{voz_path}" '
        f'-filter_complex "[1:a][2:a]amix=inputs=2:duration=first" '
        f'-c:v libx264 -t {duracion} -pix_fmt yuv420p "{video_out}" -y'
    )
    subprocess.run(cmd, shell=True)

    return guion, video_out

# Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🌌 VCPI - Generador de Entornos 3D")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Describe el entorno (ej: Cyberpunk Lab, Deep Sea, Mars)")
            btn = gr.Button("GENERAR MUNDO 3D")
        with gr.Column():
            out_v = gr.Video()
            out_t = gr.Textbox(label="Relato de la IA")

    btn.click(pipeline_maestro, [idea, gr.State(10)], [out_t, out_v])

demo.launch(share=True)
