from direct.showbase.ShowBase import ShowBase
from panda3d.core import LVector3, Fog, loadPrcFileData

class MotorVCPI:
    def __init__(self):
        # Configuración para renderizar en la nube sin ventana
        loadPrcFileData("", "window-type offscreen") 
        self.base = ShowBase()
        
    def crear_escena(self, niebla_densidad=0.1):
        # Crear Monolito
        self.monolito = self.base.loader.loadModel("models/box")
        self.monolito.setScale(1, 1, 3)
        self.monolito.reparentTo(self.base.render)
        
        # Configurar Niebla
        myFog = Fog("Fog VCPI")
        myFog.setColor(0.5, 0.5, 0.5)
        myFog.setExpDensity(niebla_densidad)
        self.base.render.setFog(myFog)
        
        # Renderizar un frame y guardarlo como imagen
        self.base.graphicsEngine.renderFrame()
        self.base.screenshot("render_vcpi.png")
        return "render_vcpi.png"
