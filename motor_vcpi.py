from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, AmbientLight, DirectionalLight, Vec4, NodePath
import os

# Forzar renderizado invisible de alta compatibilidad
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "load-display p3tinydisplay")

class MotorVCPI(ShowBase):
    def __init__(self):
        super().__init__()
        
    def crear_escena(self, estilo="cyberpunk"):
        # 1. Limpiar la mesa de trabajo
        self.render.getChildren().detach()
        self.setBackgroundColor(0.05, 0.05, 0.1) # Azul muy oscuro

        # 2. Configurar la CÁMARA (Crucial para no ver gris)
        # La movemos atrás (y=-15) y arriba (z=2) para ver la escena
        self.cam.setPos(0, -15, 2)
        self.cam.lookAt(0, 0, 0)

        # 3. CONSTRUIR EL ENTORNO
        # Suelo
        floor = self.loader.loadModel("models/box")
        floor.reparentTo(self.render)
        floor.setScale(10, 10, 0.1)
        floor.setPos(0, 0, -1)
        floor.setColor(0.1, 0.1, 0.2, 1) # Suelo azulado

        # Objeto Central (Simulando el laboratorio/figura)
        obj = self.loader.loadModel("models/smiley") # O "models/box"
        obj.reparentTo(self.render)
        obj.setPos(0, 0, 0)
        obj.setScale(2)
        
        # Color según estilo
        if "cyberpunk" in estilo:
            obj.setColor(0, 1, 1, 1) # Cyan neón
        else:
            obj.setColor(1, 0.5, 0, 1) # Naranja industrial

        # 4. ILUMINACIÓN PRO (Para dar volumen)
        # Luz de ambiente (azulada)
        alight = AmbientLight('alight')
        alight.setColor(Vec4(0.2, 0.2, 0.4, 1))
        self.render.setLight(self.render.attachNewNode(alight))

        # Luz Neón Frontal (Cyan)
        dlight = DirectionalLight('dlight')
        dlight.setColor(Vec4(0, 0.8, 1, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)

        # 5. RENDERIZAR Y GUARDAR
        self.graphicsEngine.renderFrame()
        img_path = os.path.abspath("render_final.png")
        self.screenshot(img_path, defaultFilename=False)
        return img_path
