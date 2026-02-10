import os, subprocess, torch, shutil, time
from diffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipeline
from diffusers import AutoPipelineForText2Image

# Configuración Dual-GPU Balanceada (90/50)
device_0 = "cuda:0"
device_1 = "cuda:1"

def ejecutar_produccion():
    # 1. Entrada de usuario por consola
    print("\n" + "="*50)
    prompt_usuario = input("🎬 Escribe tu idea para el video: ")
    print("="*50 + "\n")

    # Rutas
    out_mp4 = "/kaggle/working/video_final_5s.mp4"
    tmp_frames = "/kaggle/working/frames_temp"
    
    if os.path.exists(tmp_frames): shutil.rmtree(tmp_frames)
    os.makedirs(tmp_frames, exist_ok=True)

    # 2. Carga de Modelos
    print("🎨 [GPU 0] Cargando SDXL Turbo...")
    pipe_img = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    ).to(device_0)

    print("🎬 [GPU 1] Cargando SVD XT...")
    pipe_vid = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    ).to(device_1)

    # Balanceo: El VAE vive en la GPU 0 para apoyar
    pipe_vid.vae.to(device_0)
    pipe_vid.vae.to(torch.float32)
    pipe_vid.enable_sequential_cpu_offload(device=device_1)

    # 3. Generación de Imagen
    print(f"🖼️ Generando imagen base para: {prompt_usuario}")
    image = pipe_img(prompt=prompt_usuario, num_inference_steps=4).images[0].resize((512, 512))
    
    # Liberar un poco de VRAM de la GPU 0 antes del video
    del pipe_img
    torch.cuda.empty_cache()

    # 4. Generación de Video
    print("🎬 Generando 25 cuadros coherentes (5 segundos)...")
    with torch.inference_mode():
        output = pipe_vid(
            image, 
            num_frames=25, 
            motion_bucket_id=70, 
            noise_aug_strength=0.04, 
            decode_chunk_size=1
        )
    
    # 5. Guardado de frames
    print("💾 Guardando frames temporales...")
    for i, frame in enumerate(output.frames[0]):
        frame.save(f"{tmp_frames}/{i:03d}.png")

    # 6. FFMPEG (Ensamblaje Final)
    print("🎥 Compilando video MP4...")
    subprocess.run(
        f"ffmpeg -framerate 5 -i {tmp_frames}/%03d.png -c:v libx264 -pix_fmt yuv420p -movflags +faststart {out_mp4} -y",
        shell=True, capture_output=True
    )

    print(f"\n✅ ¡PRODUCCIÓN TERMINADA! \n📁 Video guardado en: {out_mp4}")
    print("Puedes descargarlo desde el panel lateral derecho de Kaggle en 'Output'.\n")

if __name__ == "__main__":
    ejecutar_produccion()
