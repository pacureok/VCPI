execute in kaggle 
# --------------------------------------------------------------------------
!pip install panda3d gradio edge-tts
import os
repo_url = "https://github.com/pacureok/VCPI.git"
if not os.path.exists('VCPI'):
    !git clone {repo_url}
%cd VCPI

!git pull origin main

!python app.py

# -----------------------------------------------------------------------------
