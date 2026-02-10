from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, AmbientLight, DirectionalLight, Vec4
import os

loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "load-display p3tinydisplay")

class MotorVCPI(ShowBase):
    def __init__(self):
        super().__init__()
        
    def crear_escena(self):
        self.render.getChildren().detach()
        
        box = self.loader.loadModel("models/box")
        box.reparentTo(self.render)
        box.setPos(0, 10, 0)
        box.setHpr(45, 45, 45)
        
        alight = AmbientLight('alight')
        alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
        self.render.setLight(self.render.attachNewNode(alight))
        
        dlight = DirectionalLight('dlight')
        dlight.setColor(Vec4(0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)
        
        self.graphicsEngine.renderFrame()
        img_name = "render_final.png"
        img_path = os.path.join(os.getcwd(), img_name)
        self.screenshot(img_path, defaultFilename=False)
        return img_path
