#include "register_types.h"
#include "core/object/class_db.h"
#include "vcpi_bridge.cpp" // Incluimos la implementación de la clase

void initialize_vcpi_bridge_module() {
    // Aquí registramos la clase para que sea visible desde GDScript y Python
    GDREGISTER_CLASS(VCPIBridge);
}

void uninitialize_vcpi_bridge_module() {
    // Limpieza si es necesaria
}
