import gradio as gr
import os
from motor_vcpi import MotorVCPI

# Iniciamos el motor 3D de Pacure AI
try:
    motor = MotorVCPI()
except:
    motor = None # Failsafe si el driver EGL falla

def pipeline_vcpi(prompt, modo):
    # 1. Render 3D (Video/Imagen)
    imagen_path = "fallback_render.png"
    if motor:
        try:
            imagen_path = motor.crear_escena(niebla_densidad=0.15)
        except Exception as e:
            print(f"Error 3D: {e}")

    # 2. Voz (TTS) - Usamos una voz mística para el estilo Pacure Pro
    voz_path = "voz.mp3"
    comando_voz = f'edge-tts --text "{prompt}" --write-media {voz_path} --voice es-MX-DaliaNeural'
    os.system(comando_voz)
    
    # 3. Log de Estado
    status = f"✅ Identidad VCPI activa. Procesado: {prompt[:30]}... en {modo}"
    
    # IMPORTANTE: Retornamos exactamente 3 valores para los 3 componentes
    return imagen_path, voz_path, status

# --- Interfaz UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌌 VCPI Multimedia Hub - Pacure AI Labs Pro")
    
    with gr.Row():
        with gr.Column():
            entrada_texto = gr.Textbox(label="Instrucción (Guion)", placeholder="Describe la escena...")
            selector_modo = gr.Dropdown(["Pelicula", "Videojuego"], label="Modo", value="Pelicula")
            boton = gr.Button("🚀 EJECUTAR IDENTIDAD", variant="primary")
        
        with gr.Column():
            resultado_img = gr.Image(label="Visión 3D (Render)")
            resultado_aud = gr.Audio(label="Voz y Audio (Dividido)")
            log_box = gr.Textbox(label="Sistema de Mensajes")

    # Aquí está el truco: conectamos los 2 inputs a los 3 outputs
    boton.click(
        fn=pipeline_vcpi, 
        inputs=[entrada_texto, selector_modo], 
        outputs=[resultado_img, resultado_aud, log_box]
    )

if __name__ == "__main__":
    # share=True es vital en Kaggle
    demo.launch(share=True, show_error=True)
