#include "register_types.h"
#include "core/object/class_db.h"
#include "core/config/engine.h"
#include "scene/main/node.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/camera_3d.h"

class VCPIBridge : public Object {
    GDCLASS(VCPIBridge, Object);

private:
    bool modo_pelicula = false;

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("configurar_modo", "es_pelicula"), &VCPIBridge::configurar_modo);
        ClassDB::bind_method(D_METHOD("instanciar_asset", "tipo", "nombre"), &VCPIBridge::instanciar_asset);
    }

public:
    void configurar_modo(bool p_pelicula) {
        modo_pelicula = p_pelicula;
        if (modo_pelicula) {
            // Ajustes para Calidad Unreal / Pacure AI Labs Pro
            Engine::get_singleton()->set_max_fps(24); // Estándar de cine
            print_line("VCPI: Modo Director Activado (Renderizado Cinematográfico)");
        } else {
            Engine::get_singleton()->set_max_fps(0); // FPS desbloqueados para juego
            print_line("VCPI: Modo Videojuego Activado (Físicas y Respuesta IA)");
        }
    }

    void instanciar_asset(String p_tipo, String p_nombre) {
        Node3D *nuevo_objeto;
        
        if (p_tipo == "CamaraCine") {
            nuevo_objeto = memnew(Camera3D);
            // Si es película, activamos efectos de profundidad de campo
            if (modo_pelicula) {
                print_line("VCPI: Configurando profundidad de campo para " + p_nombre);
            }
        } else {
            nuevo_objeto = memnew(MeshInstance3D);
        }

        nuevo_objeto->set_name(p_nombre);
        // Aquí VCPI inyectaría la malla (mesh) desde la GPU T4 de Kaggle
        print_line("VCPI: Asset creado en motor: " + p_nombre);
    }
};

void initialize_vcpi_bridge_module() {
    GDREGISTER_CLASS(VCPIBridge);
}

void uninitialize_vcpi_bridge_module() {}
