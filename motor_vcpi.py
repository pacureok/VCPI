from panda3d.core import loadPrcFileData
# ESTO ES VITAL: Configura el renderizado por software antes de importar ShowBase
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "gl-debug #t")

from direct.showbase.ShowBase import ShowBase
from panda3d.core import Fog, LVector3

class MotorVCPI(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

    def crear_escena(self, niebla_densidad=0.1):
        self.render.getChildren().detach()
        model = self.loader.loadModel("models/box")
        model.reparentTo(self.render)
        model.setPos(0, 10, 0)
        model.setScale(1, 1, 3)
        
        # Niebla estilo Backrooms/Cinematic
        fog = Fog("VCPI_Fog")
        fog.setColor(0.1, 0.1, 0.1)
        fog.setExpDensity(niebla_densidad)
        self.render.setFog(fog)
        
        self.graphicsEngine.renderFrame()
        self.screenshot("render_vcpi.png", defaultFilename=False)
        return "render_vcpi.png"
