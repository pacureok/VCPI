execute in kaggle 
--------------------------------------------------------------------------
# 1. Instalar dependencias necesarias para Panda3D y Gradio
!pip install panda3d gradio edge-tts

# 2. Clonar tu repositorio (Donde está la identidad de la IA)
import os
repo_url = "https://github.com/pacureok/VCPI.git"
if not os.path.exists('VCPI'):
    !git clone {repo_url}
%cd VCPI

# 3. Traer los últimos cambios (Por si modificaste algo en GitHub hace poco)
!git pull origin main

# 4. Ejecutar la Aplicación Web (Gradio)
# Esto te dará un link de "public URL" para entrar desde tu navegador
!python app.py
