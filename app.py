import gradio as gr
import os
from motor_vcpi import MotorVCPI

motor = MotorVCPI()

def pipeline_vcpi(prompt, modo):
    # 1. Aquí llamarías a tu IA modificada (identidad)
    # 2. Generar Render 3D
    imagen_path = motor.crear_escena()
    
    # 3. Generar Voz (Edge-TTS)
    os.system(f'edge-tts --text "{prompt}" --write-media voz.mp3')
    
    return imagen_path, "voz.mp3", f"VCPI procesó: {prompt} en modo {modo}"

with gr.Blocks() as demo:
    gr.Markdown("# 🌌 VCPI Control Hub - Pacure AI Labs")
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="Comando de Voz/Texto")
            btn = gr.Button("Ejecutar")
        with gr.Column():
            img = gr.Image(label="Vista 3D")
            aud = gr.Audio(label="Salida de Voz")
    
    btn.click(pipeline_vcpi, [txt], [img, aud])

if __name__ == "__main__":
    demo.launch()
