import gradio as gr
import os
import subprocess
import torch
import gc
import numpy as np
import soundfile as sf
from audiocraft.models import musicgen
from motor_vcpi import MotorVCPI  # Importa tu clase de Panda3D

# --- LIMPIEZA INICIAL DE MEMORIA ---
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

# --- INICIALIZACIÓN ---
print("🚀 Iniciando Cerebro Multimedia de Pacure AI Labs...")
clear_vram()

try:
    # Usamos musicgen-melody para seguir la identidad sonora
    music_model = musicgen.MusicGen.get_pretrained('facebook/musicgen-melody', device='cuda')
    motor = MotorVCPI()
    print("✅ Modelos cargados con éxito.")
except Exception as e:
    print(f"⚠️ Error en inicialización: {e}")
    motor = None

def pipeline_maestro(prompt, duracion_seg):
    clear_vram()
    
    # 1. OLLAMA: Generación de Guion (Identidad de IA)
    print("🧠 Ollama: Redactando guion...")
    # Comando blindado para evitar errores de caracteres
    guion_cmd = f"ollama run llama3 'Escribe una frase epica y corta de exploracion para: {prompt}'"
    guion = subprocess.getoutput(guion_cmd)

    # 2. MUSICGEN: Composición de 3 Minutos (Backrooms/Cinematic)
    print(f"🎵 MusicGen: Generando {duracion_seg}s de audio...")
    music_model.set_generation_params(duration=int(duracion_seg), cfg_coef=6.0)
    
    # Prompt enriquecido para la identidad musical
    music_prompt = f"Backrooms, liminal space, {prompt}, haunting dark ambient, 60bpm, high fidelity."
    res = music_model.generate([music_prompt], progress=True)
    
    audio_wav = "banda_sonora.wav"
    audio_data = res.cpu().numpy()[0, 0] # Extraer el audio correctamente
    sf.write(audio_wav, audio_data, 32000)

    # 3. EDGE-TTS: Voz Narrativa
    print("🎙️ TTS: Generando locución...")
    voz_mp3 = "narracion.mp3"
    texto_voz = guion.replace('"', '').replace('\n', ' ')[:500]
    os.system(f'edge-tts --text "{texto_voz}" --write-media {voz_mp3} --voice es-MX-DaliaNeural')

    # 4. MOTOR 3D: Renderizado de Escena
    print("🎥 Motor 3D: Capturando render...")
    render_img = "fallback.png"
    if motor:
        try:
            render_img = motor.crear_escena(niebla_densidad=0.15)
        except Exception as e:
            print(f"Error en render: {e}")

    # 5. FFMPEG: Creación del Video Final MP4
    print("🎬 FFMPEG: Ensamblando película final...")
    video_final = "VCPI_Movie_Final.mp4"
    # El comando mezcla la imagen, la música de fondo y la voz
    ffmpeg_cmd = (
        f'ffmpeg -loop 1 -i {render_img} -i {audio_wav} -i {voz_mp3} '
        f'-filter_complex "[1:a][2:a]amix=inputs=2:duration=first[aout]" '
        f'-map 0:v -map "[aout]" -c:v libx264 -t {duracion_seg} -pix_fmt yuv420p {video_final} -y'
    )
    subprocess.run(ffmpeg_cmd, shell=True)

    return guion, video_final, audio_wav

# --- INTERFAZ DE GRADIO (PÁGINA WEB) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌌 VCPI - Hub Multimedia Autónomo")
    gr.Markdown("Control de Guion, Música (3 min), Render 3D y Voces.")
    
    with gr.Row():
        with gr.Column():
            entrada_idea = gr.Textbox(label="Instrucción a la IA", placeholder="Un monolito brillante en el vacío...")
            slider_tiempo = gr.Slider(minimum=10, maximum=180, value=30, step=1, label="Duración (Segundos)")
            btn_generar = gr.Button("🚀 GENERAR UNIVERSO", variant="primary")
            
        with gr.Column():
            salida_guion = gr.Textbox(label="📜 Guion de Llama 3")
            salida_video = gr.Video(label="📽️ Película Renderizada (MP4)")
            salida_audio = gr.Audio(label="🎵 Banda Sonora (WAV)")

    # Conexión de inputs a outputs
    btn_generar.click(
        fn=pipeline_maestro,
        inputs=[entrada_idea, slider_tiempo],
        outputs=[salida_guion, salida_video, salida_audio]
    )

if __name__ == "__main__":
    # share=True permite abrir la web desde fuera de Kaggle
    demo.launch(share=True)
