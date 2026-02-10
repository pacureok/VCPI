import gradio as gr
import os
import subprocess
import torch
import numpy as np
import soundfile as sf
from audiocraft.models import musicgen
from motor_vcpi import MotorVCPI  # Importa tu clase de Panda3D

# --- INICIALIZACIÓN ---
print("📥 Cargando Modelos de IA (MusicGen & Motor 3D)...")
try:
    music_model = musicgen.MusicGen.get_pretrained('facebook/musicgen-melody', device='cuda')
    motor = MotorVCPI()
except Exception as e:
    print(f"⚠️ Error en carga inicial: {e}")
    motor = None

def pipeline_maestro(prompt, duracion):
    # 1. OLLAMA: Guion Cinematográfico
    print("🧠 Generando Guion...")
    guion_cmd = f"ollama run llama3 'Escribe un guion corto y oscuro para: {prompt}'"
    guion = subprocess.getoutput(guion_cmd)

    # 2. MUSICGEN: Audio de 3 Minutos (o lo seleccionado)
    print("🎵 Componiendo Música...")
    music_model.set_generation_params(duration=int(duracion), cfg_coef=5.5)
    res = music_model.generate([f"Backrooms, liminal space, {prompt}, dark ambient"], progress=True)
    audio_wav = "temp_music.wav"
    sf.write(audio_wav, np.squeeze(res.cpu().numpy()), 32000)

    # 3. EDGE-TTS: Voz de la IA
    print("🎙️ Generando Narración...")
    voz_mp3 = "temp_voz.mp3"
    texto_limpio = guion.replace('"', '').replace('\n', ' ')[:400]
    os.system(f'edge-tts --text "{texto_limpio}" --write-media {voz_mp3} --voice es-MX-DaliaNeural')

    # 4. MOTOR 3D: Render de Escena
    print("🎥 Renderizando 3D...")
    render_img = "fallback.png"
    if motor:
        render_img = motor.crear_escena(niebla_densidad=0.12)

    # 5. FFMPEG: Ensamblaje de Película Final
    video_final = "Pelicula_VCPI_Final.mp4"
    # Mezcla audio de fondo con voz y pone la imagen de fondo
    ffmpeg_cmd = (
        f'ffmpeg -loop 1 -i {render_img} -i {audio_wav} -i {voz_mp3} '
        f'-filter_complex "[1:a][2:a]amix=inputs=2:duration=first[a]" '
        f'-map 0:v -map "[a]" -c:v libx264 -t {duracion} -pix_fmt yuv420p {video_final} -y'
    )
    os.system(ffmpeg_cmd)

    return guion, video_final, audio_wav

# --- INTERFAZ GRADIO ---
with gr.Blocks(title="VCPI Pacure Pro") as demo:
    gr.Markdown("# 🌌 VCPI: Sistema Multimedia Autónomo")
    with gr.Row():
        with gr.Column():
            idea = gr.Textbox(label="Idea Central", placeholder="Ej: Un pasillo infinito...")
            tiempo = gr.Slider(10, 180, value=30, label="Segundos de Video/Música")
            btn = gr.Button("🎬 GENERAR PELÍCULA", variant="primary")
        with gr.Column():
            out_guion = gr.Textbox(label="📜 Guion")
            out_video = gr.Video(label="📽️ Película Final (MP4)")
            out_audio = gr.Audio(label="🎵 Banda Sonora Original")

    btn.click(pipeline_maestro, [idea, tiempo], [out_guion, out_video, out_audio])

if __name__ == "__main__":
    demo.launch(share=True)
