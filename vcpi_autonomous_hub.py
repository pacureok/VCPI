import gradio as gr
import subprocess
import os
import torch
import torchaudio
import soundfile as sf
import numpy as np

# --- 1. FUNCIÓN OLLAMA (GUIONISTA) ---
def generar_guion(descripcion):
    print("🧠 Ollama pensando guion...")
    # Llamamos a tu identidad 'vcpi_architect' que creamos antes
    cmd = f"ollama run vcpi_architect 'Escribe un guion cinematográfico breve basado en: {descripcion}'"
    guion = subprocess.check_output(cmd, shell=True).decode('utf-8')
    return guion

# --- 2. FUNCIÓN MUSICGEN (MÚSICA BACKROOMS) ---
def generar_musica_3min(prompt_audio):
    print("⏳ Generando 3 minutos de vacío (Backrooms)...")
    # Usamos los parámetros que ya tienes configurados
    model.set_generation_params(duration=180, use_sampling=True, top_k=250, cfg_coef=5.5)
    
    # Generación pura de texto (sin archivo base para evitar errores de carga)
    res = model.generate([prompt_audio], progress=True)
    res_cpu = np.squeeze(res.cpu().numpy())
    ruta_audio = 'musica_backrooms.wav'
    sf.write(ruta_audio, res_cpu, 32000)
    return ruta_audio

# --- 3. PIPELINE INTEGRADO ---
def produccion_total(idea_usuario):
    # Paso A: Guion
    guion = generar_guion(idea_usuario)
    
    # Paso B: Música (Usando tu prompt de Backrooms)
    prompt_backrooms = (
        'Backrooms liminal space aesthetic, haunting dark ambient, low-fidelity VHS tape hiss. '
        'Atmosphere: unsettling, depressive, nostalgic, 60bpm, cinematic horror.'
    )
    archivo_musica = generar_musica_3min(prompt_backrooms)
    
    # Paso C: Render 3D (Usando Panda3D o Godot binario que ya tienes)
    # Aquí el motor 3D genera 'render_3d.png' (o .mp4 si tienes ffmpeg)
    # Por ahora, usamos el motor de Panda3D que configuramos arriba
    render_path = motor.crear_escena(niebla_densidad=0.2) 
    
    return guion, render_path, archivo_musica

# --- 4. INTERFAZ GRADIO ---
with gr.Blocks(title="VCPI Cinema Studio") as demo:
    gr.Markdown("# 🎬 VCPI Cinema Studio: Guion, 3D y Música (3 Min)")
    
    with gr.Row():
        entrada = gr.Textbox(label="Idea de la Película", placeholder="Un explorador perdido en el nivel 0...")
        btn = gr.Button("🚀 PRODUCCIÓN TOTAL", variant="primary")
        
    with gr.Row():
        out_guion = gr.Textbox(label="Guion Generado (Ollama)")
        out_render = gr.Image(label="Visualización 3D Real")
        out_audio = gr.Audio(label="Banda Sonora (3 Min)")

    btn.click(produccion_total, inputs=[entrada], outputs=[out_guion, out_render, out_audio])

demo.launch(share=True)
