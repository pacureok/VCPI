execute in kaggle 
--------------------------------------------------------------------------
```bash
# 1. Instalación de dependencias de sistema
!apt-get update && apt-get install -y ffmpeg zstd libosmesa6-dev

# 2. Instalación de librerías de Python (Orden corregido para evitar errores de 'av')
!pip install --only-binary=:all: av==12.3.0
!pip install panda3d edge-tts gradio soundfile xformers==0.0.23.post1

# 3. Instalación de Audiocraft (Modo Seguro)
!pip install --no-deps git+https://github.com/facebookresearch/audiocraft
!pip install flashy>=0.0.1 hydra-core>=1.1 julius num2words omegaconf pesq pystoi torchdiffeq torchmetrics

# 4. Instalación y arranque de Ollama
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess
import time
import os

OLLAMA_PATH = "/usr/local/bin/ollama"
subprocess.Popen([OLLAMA_PATH, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(10)
!{OLLAMA_PATH} pull llama3

# 5. Clonar tu repositorio
if os.path.exists('/kaggle/working/VCPI'):
    import shutil
    shutil.rmtree('/kaggle/working/VCPI')
%cd /kaggle/working/
!git clone https://github.com/pacureok/VCPI.git
%cd VCPI

# 6. Lanzamiento
os.environ['OLLAMA_BIN'] = OLLAMA_PATH
!python app.py
