Licencia de Uso Propietario: VCPI & Pacure Labs
Versión v10+ - 2026

Por la presente, se establece que el software (código fuente), los pesos del modelo y los activos digitales contenidos en este repositorio son propiedad exclusiva de Pacure Labs. El uso de este material está sujeto a las siguientes restricciones:

🚫 Restricciones Estrictas
Prohibición de Modificación: No se permite la alteración, edición, transformación o creación de obras derivadas del código fuente o de la arquitectura del modelo.

Prohibición de Distribución: No se permite la redistribución, sublicenciamiento, alquiler o préstamo del código o del modelo a terceros, ya sea de forma gratuita o comercial.

Prohibición de Ingeniería Inversa: No se permite descompilar o intentar extraer la lógica interna del modelo para crear versiones alternativas.

✅ Derechos de Comercialización
Venta del Producto Final: Se autoriza al usuario a vender los archivos generados por el software (ej. videos .mp4, imágenes o resultados procesados).

Atribución Obligatoria: Para cualquier uso comercial o exhibición pública del producto generado, es requisito indispensable incluir de forma visible el siguiente crédito:

"Generado con motor VCPI por Pacure Labs"

⚖️ Incumplimiento
Cualquier violación a estos términos resultará en la revocación inmediata de la licencia de uso y dará lugar a las acciones legales correspondientes bajo las leyes de propiedad intelectual internacionales
------------------------------------------------
ejuctar en kaggle
```bash
import os, torch, subprocess, shutil

# 1. Limpieza total
os.chdir('/kaggle/working')
if os.path.exists('VCPI'): shutil.rmtree('VCPI')
!rm -rf frames_temp produccion_final.mp4
torch.cuda.empty_cache()

# 2. Instalación con versiones fijas (Evita el error de transformers 5.1.0)
print("📦 Instalando versiones de máxima compatibilidad...")
!pip install -qU "transformers>=4.41.0,<4.45.0" "diffusers>=0.30.0" "accelerate>=0.33.0"
!pip install -qU bitsandbytes gradio

# 3. Clonar repositorio
!git clone https://github.com/pacureok/VCPI.git
os.chdir('VCPI')

# 4. Configuración Multi-GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

# 5. Ejecutar
print("🚀 Lanzando VCPI (MP4 Directo)...")
!python app.py
