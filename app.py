import gradio as gr
import os
import subprocess
import torch
import soundfile as sf
import numpy as np

# --- INTENTO DE IMPORTACIÓN ROBUSTA ---
try:
    from audiocraft.models import musicgen
    AUDIO_ENGINE = "audiocraft"
except ImportError:
    AUDIO_ENGINE = "none"
    print("⚠️ MusicGen no disponible, el audio será omitido o genérico.")

try:
    from motor_vcpi import MotorVCPI
    motor = MotorVCPI()
except ImportError:
    motor = None
    print("⚠️ Motor 3D no detectado.")

def pipeline_maestro(prompt, duracion):
    # 1. TEXTO (Guion vía Ollama)
    print("📝 Generando guion...")
    ollama_bin = os.getenv('OLLAMA_BIN', '/usr/local/bin/ollama')
    try:
        guion = subprocess.check_output([ollama_bin, "run", "llama3", f"Escribe una frase corta de 10 palabras sobre: {prompt}"], timeout=30).decode('utf-8')
    except:
        guion = f"Explorando el sector: {prompt}. Anomalía detectada."

    # 2. VOCES (Edge-TTS)
    print("🎙️ Generando voz...")
    voz_path = "narracion.mp3"
    subprocess.run(["edge-tts", "--text", guion, "--write-media", voz_path, "--voice", "es-MX-DaliaNeural"])

    # 3. MÚSICA (MusicGen)
    print("🎵 Generando música...")
    musica_path = "ambiente.wav"
    if AUDIO_ENGINE == "audiocraft":
        try:
            model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small')
            model.set_generation_params(duration=int(duracion))
            res = model.generate([f"Dark ambient, cinematic, {prompt}"])
            audio_data = res[0, 0].cpu().numpy()
            sf.write(musica_path, audio_data, 32000)
        except:
            # Generar silencio si falla para no romper el video
            sf.write(musica_path, np.zeros(32000 * int(duracion)), 32000)
    else:
        sf.write(musica_path, np.zeros(32000 * int(duracion)), 32000)

    # 4. VIDEO (Render 3D)
    print("🎥 Generando imagen 3D...")
    if motor:
        img_path = motor.crear_escena()
    else:
        img_path = "fallback.png" # Asegúrate de tener una imagen base o generarla

    # 5. ENSAMBLAJE FINAL (FFMPEG)
    print("🎬 Ensamblando video final...")
    video_final = "VCPI_Resultado.mp4"
    # Une imagen + música + voz
    ffmpeg_cmd = (
        f'ffmpeg -loop 1 -i {img_path} -i {musica_path} -i {voz_path} '
        f'-filter_complex "[1:a][2:a]amix=inputs=2:duration=first" '
        f'-c:v libx264 -t {duracion} -pix_fmt yuv420p {video_final} -y'
    )
    subprocess.run(ffmpeg_cmd, shell=True)

    return guion, video_final, musica_path

# --- INTERFAZ ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🌌 VCPI: Generador de Cine IA")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Idea de la escena")
            seg = gr.Slider(5, 60, value=15, label="Duración")
            btn = gr.Button("🚀 GENERAR TODO", variant="primary")
        with gr.Column():
            res_txt = gr.Textbox(label="Guion (Ollama)")
            res_vid = gr.Video(label="Video Final (3D + Voces + Música)")
            res_aud = gr.Audio(label="Banda Sonora")

    btn.click(pipeline_maestro, [idea, seg], [res_txt, res_vid, res_aud])

demo.launch(share=True)
