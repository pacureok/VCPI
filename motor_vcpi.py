from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, AmbientLight, DirectionalLight, Vec4
import os

# Forzar renderizado "Offscreen"
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "load-display p3tinydisplay")

class MotorVCPI(ShowBase):
    def __init__(self):
        super().__init__()
        
    def crear_escena(self):
        self.render.getChildren().detach() # Limpiar
        
        # Crear un objeto (Caja)
        box = self.loader.loadModel("models/box")
        box.reparentTo(self.render)
        box.setPos(0, 10, 0)
        box.setHpr(45, 45, 45)
        
        # --- ILUMINACIÓN (Vital para que no sea negro) ---
        # Luz Ambiental (luz general suave)
        alight = AmbientLight('alight')
        alight.setColor(Vec4(0.4, 0.4, 0.4, 1))
        self.render.setLight(self.render.attachNewNode(alight))
        
        # Luz Direccional (como el sol, crea sombras y volumen)
        dlight = DirectionalLight('dlight')
        dlight.setColor(Vec4(0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)
        
        # Captura
        self.graphicsEngine.renderFrame()
        img_path = os.path.abspath("render_final.png")
        self.screenshot(img_path, defaultFilename=False)
        return img_path
