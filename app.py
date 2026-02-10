import gradio as gr
import os
import subprocess
import time
import soundfile as sf
import numpy as np
from motor_vcpi import MotorVCPI

def pipeline_maestro(prompt, duracion):
    # 1. Configuración de Rutas Absolutas
    # Esto evita que el sistema se pierda si la carpeta cambia
    base_dir = os.path.dirname(os.path.abspath(__file__))
    voz_path = os.path.join(base_dir, "voz.mp3")
    musica_path = os.path.join(base_dir, "musica.wav")
    video_out = os.path.join(base_dir, "VCPI_Final.mp4")
    
    ollama_bin = os.getenv('OLLAMA_BIN', '/usr/local/bin/ollama')

    # 2. Generación de Guion (Ollama)
    print(f"📝 Solicitando guion para: {prompt}")
    try:
        # Limitamos a 10 palabras para que la locución sea rápida
        guion = subprocess.check_output(
            [ollama_bin, "run", "llama3", f"Escribe una frase de 10 palabras sobre: {prompt}"], 
            timeout=30
        ).decode('utf-8').strip()
    except Exception as e:
        print(f"⚠️ Error Ollama: {e}")
        guion = f"Iniciando simulación de laboratorio para {prompt}. Protocolo activo."

    # 3. Generación de Voz (Edge-TTS)
    print("🎙️ Generando locución...")
    try:
        # Usamos subprocess.run para asegurar que el archivo se escriba en disco
        subprocess.run([
            "edge-tts", 
            "--text", guion, 
            "--write-media", voz_path, 
            "--voice", "es-MX-DaliaNeural"
        ], check=True)
    except Exception as e:
        print(f"⚠️ Error Voz: {e}")

    # 4. Generación de Música (Failsafe)
    print("🎵 Creando banda sonora...")
    # Por ahora creamos un fondo neutro para asegurar que FFMPEG no falle
    # (Puedes integrar MusicGen aquí cuando la memoria de Kaggle lo permita)
    samplerate = 32000
    silencio = np.zeros(samplerate * int(duracion))
    sf.write(musica_path, silencio, samplerate)

    # 5. Renderizado 3D (Panda3D)
    print("🎥 Renderizando escena 3D...")
    try:
        motor = MotorVCPI()
        img_render = motor.crear_escena() 
    except Exception as e:
        print(f"⚠️ Error 3D: {e}")
        # Crear un fallback negro si falla el render
        img_render = os.path.join(base_dir, "fallback.png")
        if not os.path.exists(img_render):
            from PIL import Image
            Image.new('RGB', (800, 600), (20, 20, 20)).save(img_render)

    # 6. Espera de Seguridad (Sincronización de Disco)
    time.sleep(2)

    # 7. Ensamblaje Final (FFMPEG)
    print("🎬 Ensamblando video final...")
    if os.path.exists(voz_path) and os.path.exists(img_render):
        # El comando mezcla la música de fondo con la voz
        ffmpeg_cmd = (
            f'ffmpeg -loop 1 -i "{img_render}" -i "{musica_path}" -i "{voz_path}" '
            f'-filter_complex "[1:a][2:a]amix=inputs=2:duration=first" '
            f'-c:v libx264 -t {duracion} -pix_fmt yuv420p "{video_out}" -y'
        )
        subprocess.run(ffmpeg_cmd, shell=True)
    else:
        return f"Error: Archivos faltantes. Guion: {guion}", None

    return guion, video_out

# --- INTERFAZ DE GRADIO ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🌌 VCPI v4.0 - Multimodal Lab")
    
    with gr.Row():
        with gr.Column():
            input_prompt = gr.Textbox(
                label="¿Qué quieres ver en el laboratorio?", 
                placeholder="Ej: Cyberpunk robot assembling a crystal"
            )
            input_time = gr.Slider(5, 20, value=10, label="Duración (segundos)")
            generate_btn = gr.Button("🚀 GENERAR MULTIMODAL", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="Guion de la IA")
            output_video = gr.Video(label="Producción Final")

    generate_btn.click(
        fn=pipeline_maestro,
        inputs=[input_prompt, input_time],
        outputs=[output_text, output_video]
    )

if __name__ == "__main__":
    demo.launch(share=True)
