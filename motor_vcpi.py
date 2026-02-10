from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, AntialiasAttrib
import os

# Forzar renderizado "Offscreen" (Sin ventana)
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "load-display p3tinydisplay") # Motor de software alternativo

class MotorVCPI(ShowBase):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor(0.1, 0.1, 0.1) # Fondo gris oscuro (no negro total)
        
    def crear_escena(self):
        # Limpiar escena previa
        self.render.getChildren().detach()
        
        # Crear un objeto de prueba (Caja) para verificar que no esté negro
        box = self.loader.loadModel("models/box")
        box.reparentTo(self.render)
        box.setPos(0, 5, 0)
        box.setScale(1)
        box.setHpr(45, 45, 45)
        
        # Luces (Indispensables para que no se vea negro)
        from panda3d.core import AmbientLight, DirectionalLight
        alight = AmbientLight('alight')
        alight.setColor((0.5, 0.5, 0.5, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
        # Renderizar un frame y guardar
        self.graphicsEngine.renderFrame()
        img_path = "render_3d.png"
        self.screenshot(img_path, defaultFilename=False)
        return os.path.abspath(img_path)
